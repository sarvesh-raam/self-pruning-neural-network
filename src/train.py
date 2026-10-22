import os
import time
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .model import SelfPruningNet

# Per-channel CIFAR-10 statistics
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class TensorLoader:
    """
    Minimal in-memory batch iterator.

    CIFAR-10 is only 150 MB as uint8, so the whole split lives on the target
    device and augmentation (random crop + horizontal flip) runs on batched
    tensors. This removes the PIL / DataLoader-worker bottleneck that otherwise
    dominates the runtime of a small MLP.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                 shuffle: bool, augment: bool, device: str):
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        # (N, 32, 32, 3) uint8 -> normalized (N, 3, 32, 32) float on device
        x = images.permute(0, 3, 1, 2).float().div_(255.0)
        self.images = ((x - mean) / std).to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.device = device

    def __len__(self) -> int:
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        flip = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        # Random 32x32 crop of the 4-pixel zero-padded batch (one offset per batch keeps it cheap)
        padded = F.pad(x, (4, 4, 4, 4))
        dx, dy = torch.randint(0, 9, (2,)).tolist()
        return padded[:, :, dy:dy + 32, dx:dx + 32]

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            x = self.images[idx]
            if self.augment:
                x = self._augment(x)
            yield x, self.labels[idx]


def load_cifar10(data_dir: str = "./data", batch_size: int = 256, device: str = "cpu",
                 download: bool = True) -> tuple[TensorLoader, TensorLoader]:
    """Return (train_loader, test_loader) for CIFAR-10."""
    os.makedirs(data_dir, exist_ok=True)
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download)
    train_loader = TensorLoader(torch.from_numpy(train.data), torch.tensor(train.targets),
                                batch_size, shuffle=True, augment=True, device=device)
    test_loader = TensorLoader(torch.from_numpy(test.data), torch.tensor(test.targets),
                               1000, shuffle=False, augment=False, device=device)
    return train_loader, test_loader


@dataclass
class TrainConfig:
    lambda_val: float = 1e-5     # weight of the L1 gate penalty (sum over ~1.7M gates)
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3             # learning rate for weights and biases
    gate_lr: float = 5e-2        # learning rate for gate_scores (see Trainer docstring)
    warmup_epochs: int = 2       # lambda is ramped linearly from 0 over these epochs
    threshold: float = 1e-2      # gate value below which a weight counts as pruned
    seed: int = 42
    device: str = field(default_factory=resolve_device)


class Trainer:
    """
    Trains a SelfPruningNet with  total_loss = CE + lambda * L1(gates).

    The gates get their own optimizer parameter group with a higher learning
    rate. Adam's update size is roughly `lr` per step regardless of the gradient
    scale, so with lr = 1e-3 a gate score can move at most ~1e-3 per step. Going
    from an open gate (sigmoid ~ 0.9) to a pruned one (sigmoid < 0.01) needs a
    change of ~6.6 in the score, i.e. more steps than a whole training run.
    That is why the original version reported 0% sparsity for every lambda.
    """

    def __init__(self, model: SelfPruningNet, config: TrainConfig):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        gate_ids = {id(p) for p in model.gate_parameters()}
        weights = [p for p in model.parameters() if id(p) not in gate_ids]
        self.optimizer = torch.optim.Adam([
            {"params": weights, "lr": config.lr},
            {"params": model.gate_parameters(), "lr": config.gate_lr},
        ])
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.train_loader: Optional[TensorLoader] = None
        self.test_loader: Optional[TensorLoader] = None

    def set_data(self, train_loader: TensorLoader, test_loader: TensorLoader) -> None:
        self.train_loader, self.test_loader = train_loader, test_loader

    def current_lambda(self, epoch: int) -> float:
        """Linear warm-up of lambda so the network learns useful features before pruning starts."""
        if self.config.warmup_epochs <= 0:
            return self.config.lambda_val
        return self.config.lambda_val * min(1.0, epoch / self.config.warmup_epochs)

    def train_epoch(self, lambda_val: float) -> Dict[str, float]:
        self.model.train()
        sums = {"class_loss": 0.0, "sparsity_loss": 0.0, "total_loss": 0.0}
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            class_loss = self.criterion(self.model(inputs), targets)
            # Not detached: the penalty's gradient must reach gate_scores
            sparsity_loss = self.model.sparsity_loss()
            total_loss = class_loss + lambda_val * sparsity_loss
            total_loss.backward()
            self.optimizer.step()

            sums["class_loss"] += class_loss.item()
            sums["sparsity_loss"] += sparsity_loss.item()
            sums["total_loss"] += total_loss.item()
        return {k: v / len(self.train_loader) for k, v in sums.items()}

    @torch.no_grad()
    def accuracy(self) -> float:
        self.model.eval()
        correct = total = 0
        for inputs, targets in self.test_loader:
            correct += (self.model(inputs).argmax(1) == targets).sum().item()
            total += targets.numel()
        return 100.0 * correct / total

    def evaluate(self) -> tuple[float, float]:
        """Test accuracy and sparsity (%) of the model in its current state."""
        return self.accuracy(), self.model.get_model_sparsity(self.config.threshold)

    def train(self, log=print) -> Dict[str, List[float]]:
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError("Data not set. Call set_data() first.")
        cfg = self.config
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

        history: Dict[str, List[float]] = {k: [] for k in
                                           ("class_loss", "sparsity_loss", "total_loss", "accuracy", "sparsity", "lambda")}
        num_gates = self.model.num_gates()
        log(f"Training {cfg.epochs} epochs | lambda = {cfg.lambda_val} | device = {self.device}")
        for epoch in range(1, cfg.epochs + 1):
            start = time.perf_counter()
            lam = self.current_lambda(epoch)
            losses = self.train_epoch(lam)
            self.scheduler.step()
            acc, spars = self.evaluate()

            for k, v in losses.items():
                history[k].append(v)
            history["accuracy"].append(acc)
            history["sparsity"].append(spars)
            history["lambda"].append(lam)
            log(f"Epoch {epoch:02d}/{cfg.epochs} | CE {losses['class_loss']:.4f} | "
                f"mean gate {losses['sparsity_loss'] / num_gates:.4f} | acc {acc:.2f}% | "
                f"sparsity {spars:.2f}% | {time.perf_counter() - start:.1f}s")
        return history

    def finalize(self) -> Dict[str, float]:
        """
        Hard-prune every gate below the threshold and measure the truly sparse model.

        Accuracy before and after are both reported, so it is visible that the
        pruned connections really were unnecessary.
        """
        soft_acc, sparsity = self.evaluate()
        self.model.prune(self.config.threshold)
        pruned_acc = self.accuracy()
        total = self.model.num_gates()
        remaining = int(sum(m.mask.sum().item() for _, m in self.model.prunable_layers()))
        return {
            "lambda": self.config.lambda_val,
            "accuracy_soft": soft_acc,
            "accuracy": pruned_acc,
            "sparsity": sparsity,
            "weights_total": total,
            "weights_remaining": remaining,
            "compression": total / max(remaining, 1),
            "layer_sparsity": self.model.get_layer_sparsity(self.config.threshold),
        }

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_config": self.model.config,
            "train_config": asdict(self.config),
            "state_dict": self.model.state_dict(),
            "metrics": metrics or {},
        }, path)


def load_model(path: str, device: str = "cpu") -> tuple[SelfPruningNet, Dict]:
    """Rebuild a SelfPruningNet from a checkpoint written by `Trainer.save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SelfPruningNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
