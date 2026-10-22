import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A linear layer that learns to prune its own weights during training.

    Every weight w_ij is paired with a learnable score s_ij. The forward pass uses

        gates          = sigmoid(gate_scores)            # in (0, 1)
        pruned_weights = weight * gates * mask
        output         = x @ pruned_weights^T + bias

    An L1 penalty on `gates` (applied by the trainer) pushes unimportant gates
    towards 0. Because a sigmoid never reaches exactly 0, a gate below the
    threshold is only "soft" pruned. Calling `prune(threshold)` turns that into
    a hard prune by zeroing the `mask` buffer, so the weight contributes exactly
    nothing and the reported accuracy is the accuracy of the truly sparse model.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gate_init: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_init = gate_init

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Learnable gate scores, same shape as weight, registered as a parameter
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Hard-prune mask (1 = kept, 0 = pruned). A buffer, so it is saved in the
        # state_dict and moved with .to(device) but never touched by the optimizer.
        self.register_buffer("mask", torch.ones(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias like nn.Linear, and gates to sigmoid(gate_init)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # gate_init = 2.0 gives sigmoid(2) ~ 0.88: every connection starts open,
        # but close enough to the linear region of the sigmoid that the optimizer
        # can actually close a gate within a normal training budget.
        nn.init.constant_(self.gate_scores, self.gate_init)
        self.mask.fill_(1.0)

    def gates(self) -> torch.Tensor:
        """Differentiable gate values in (0, 1), with hard-pruned entries forced to 0."""
        return torch.sigmoid(self.gate_scores) * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradients flow into both `weight` and `gate_scores` through this product
        return F.linear(x, self.weight * self.gates(), self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Current gate values, detached from the graph."""
        return self.gates().detach()

    @torch.no_grad()
    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage (0-100) of gates below `threshold`."""
        return (self.get_gates() < threshold).float().mean().item() * 100.0

    @torch.no_grad()
    def prune(self, threshold: float = 1e-2) -> int:
        """Permanently remove every connection whose gate is below `threshold`. Returns how many."""
        pruned = self.get_gates() < threshold
        self.mask[pruned] = 0.0
        return int(pruned.sum().item())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
