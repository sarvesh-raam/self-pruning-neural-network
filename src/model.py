import torch
import torch.nn as nn
from typing import Dict, Iterator, Sequence
from .prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    A feed-forward self-pruning network for CIFAR-10.

    Architecture (default): 3072 -> 512 -> 256 -> 128 -> 10, every linear layer a
    PrunableLinear, ReLU activations and Dropout in between.

    There is deliberately no BatchNorm after the gated layers. BatchNorm makes a
    layer's output invariant to the scale of its weights, so the network could
    shrink *every* gate uniformly (satisfying the L1 penalty) while BatchNorm
    silently rescales the activations back. Gates would look "pruned" without
    any connection actually being removed.
    """

    def __init__(
        self,
        in_features: int = 3 * 32 * 32,
        hidden: Sequence[int] = (512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
        gate_init: float = 2.0,
    ):
        super().__init__()
        self.config = dict(in_features=in_features, hidden=list(hidden), num_classes=num_classes,
                           dropout=dropout, gate_init=gate_init)

        self.net = nn.Sequential()
        self.net.add_module("flatten", nn.Flatten())
        prev = in_features
        for i, width in enumerate(hidden, start=1):
            self.net.add_module(f"fc{i}", PrunableLinear(prev, width, gate_init=gate_init))
            self.net.add_module(f"relu{i}", nn.ReLU(inplace=True))
            self.net.add_module(f"drop{i}", nn.Dropout(dropout))
            prev = width
        # Output layer produces raw logits
        self.net.add_module(f"fc{len(hidden) + 1}", PrunableLinear(prev, num_classes, gate_init=gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def prunable_layers(self) -> Iterator[tuple[str, PrunableLinear]]:
        for name, module in self.net.named_children():
            if isinstance(module, PrunableLinear):
                yield name, module

    def gate_parameters(self) -> list[nn.Parameter]:
        return [m.gate_scores for _, m in self.prunable_layers()]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Gates are sigmoid outputs and therefore always positive, so the L1 norm is
        simply their sum. With ~1.7M gates this sum is large, which is why the
        useful lambda values are small (around 1e-6 to 1e-5).
        """
        return sum(m.gates().sum() for _, m in self.prunable_layers())

    def num_gates(self) -> int:
        return sum(m.gate_scores.numel() for _, m in self.prunable_layers())

    @torch.no_grad()
    def get_all_gates(self) -> torch.Tensor:
        """All gate values concatenated into one flat tensor (for histograms)."""
        return torch.cat([m.get_gates().flatten() for _, m in self.prunable_layers()])

    @torch.no_grad()
    def get_model_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of all gates in the model below `threshold`."""
        return (self.get_all_gates() < threshold).float().mean().item() * 100.0

    @torch.no_grad()
    def get_layer_sparsity(self, threshold: float = 1e-2) -> Dict[str, float]:
        """Sparsity percentage of each PrunableLinear layer."""
        return {name: m.get_sparsity(threshold) for name, m in self.prunable_layers()}

    @torch.no_grad()
    def prune(self, threshold: float = 1e-2) -> int:
        """Hard-prune every layer. Returns the total number of removed connections."""
        return sum(m.prune(threshold) for _, m in self.prunable_layers())
