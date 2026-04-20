import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Tuple, List, Any
from tqdm import tqdm
from .model import SelfPruningNet
from .prunable_layer import PrunableLinear

class Trainer:
    """Handles the training and evaluation of the SelfPruningNet."""
    def __init__(self, model: SelfPruningNet, lambda_val: float, device: str, lr: float = 1e-3):
        self.model = model.to(device)
        self.lambda_val = lambda_val
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer updates ALL parameters, importantly including the gate_scores
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_loader = None
        self.test_loader = None

    def load_cifar10(self, batch_size: int = 256) -> None:
        """Loads CIFAR-10 dataset with augmentation for training and normalization for both."""
        # Mean and STD for CIFAR-10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=2)

    def train_epoch(self) -> Dict[str, float]:
        """
        Trains the model for one epoch.
        Computes both Classification Loss and the custom Sparsity Loss.
        """
        self.model.train()
        epoch_class_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_total_loss = 0.0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            # 1. Standard Classification Loss
            class_loss = self.criterion(outputs, targets)
            
            # 2. Sparsity Loss: L1 norm of all sigmoid gate values
            # We compute it by summing sigmoid(gate_scores) over all PrunableLinear layers
            sparsity_loss = torch.tensor(0.0, device=self.device)
            for module in self.model.modules():
                if isinstance(module, PrunableLinear):
                    # We must NOT detach here, we want gradients to flow through gate_scores
                    gates = torch.sigmoid(module.gate_scores)
                    sparsity_loss += torch.sum(gates)
            
            # 3. Total Loss
            total_loss = class_loss + (self.lambda_val * sparsity_loss)
            
            total_loss.backward()
            self.optimizer.step()
            
            epoch_class_loss += class_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_total_loss += total_loss.item()
            
        num_batches = len(self.train_loader)
        return {
            "class_loss": epoch_class_loss / num_batches,
            "sparsity_loss": epoch_sparsity_loss / num_batches,
            "total_loss": epoch_total_loss / num_batches
        }

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates the model on the test set, returning accuracy and model sparsity."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = 100.0 * correct / total
        sparsity = self.model.get_model_sparsity()
        
        return accuracy, sparsity

    def train(self, epochs: int = 25) -> Dict[str, List[float]]:
        """Full training loop over multiple epochs."""
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError("Data not loaded. Call load_cifar10() first.")
            
        history = {
            "class_loss": [], "sparsity_loss": [], "total_loss": [],
            "accuracy": [], "sparsity": []
        }
        
        print(f"Starting training for {epochs} epochs with lambda = {self.lambda_val} on {self.device}")
        
        for epoch in range(1, epochs + 1):
            losses = self.train_epoch()
            acc, spars = self.evaluate()
            
            history["class_loss"].append(losses["class_loss"])
            history["sparsity_loss"].append(losses["sparsity_loss"])
            history["total_loss"].append(losses["total_loss"])
            history["accuracy"].append(acc)
            history["sparsity"].append(spars)
            
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Total Loss: {losses['total_loss']:.4f} | "
                  f"Class Loss: {losses['class_loss']:.4f} | "
                  f"Acc: {acc:.2f}% | Sparsity: {spars:.2f}%")
                  
        return history

    def save_checkpoint(self, path: str) -> None:
        """Saves model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Loads model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Checkpoint loaded from {path}")
