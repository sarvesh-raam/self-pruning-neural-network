import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A custom PyTorch linear layer that learns to prune its own weights during training.
    
    This layer introduces a second parameter, `gate_scores`, which has the exact same
    shape as the standard `weight` parameter. During the forward pass, a sigmoid
    function squashes `gate_scores` into values between 0 and 1 (gates). The actual
    weights used for the linear transformation are the element-wise product of the
    standard weights and these gates.
    
    When a gate value is pushed to 0 (typically via an L1 sparsity penalty), the
    corresponding weight is effectively "pruned" from the network.
    """
    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores, same shape as weight
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize weights, biases, and gate scores."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores to a positive value (e.g., 3.0) so the initial sigmoid(gates) is ~0.95.
        # This means initially all weights are mostly active before pruning begins.
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the dynamic gating mechanism.
        
        Math:
            gates = sigmoid(gate_scores)
            pruned_weights = weight * gates
            output = x @ pruned_weights^T + bias
        """
        # Apply sigmoid to gate_scores to get gates in range (0, 1)
        gates = torch.sigmoid(self.gate_scores)
        
        # Multiply standard weights by gates (element-wise)
        # Gradients flow through BOTH weight and gate_scores
        pruned_weights = self.weight * gates
        
        # Perform standard linear operation with the pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (between 0 and 1) without tracking gradients."""
        return torch.sigmoid(self.gate_scores).detach()

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Calculates the percentage of gates that are below the given threshold.
        
        Args:
            threshold (float): The value below which a gate is considered "pruned".
        
        Returns:
            float: Sparsity percentage (0.0 to 100.0).
        """
        gates = self.get_gates()
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return (pruned_count / total_count) * 100.0
