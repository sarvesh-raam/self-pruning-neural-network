import torch
import torch.nn as nn
from typing import Dict
from .prunable_layer import PrunableLinear

class SelfPruningNet(nn.Module):
    """
    A Self-Pruning Neural Network for CIFAR-10 classification.
    
    Architecture:
        Input (3x32x32 = 3072) -> 512 -> 256 -> 128 -> 10 (Output)
        
    Each linear layer is a PrunableLinear layer, which learns to prune its own
    weights. Activations are ReLU, and we use BatchNorm for stability and Dropout
    for standard regularization, complementing the L1 sparsity penalty.
    """
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        
        # CIFAR-10 images are 3 channels, 32x32 pixels
        self.flatten = nn.Flatten()
        
        # Layer 1: 3072 -> 512
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        
        # Layer 2: 512 -> 256
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        
        # Layer 3: 256 -> 128
        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)
        
        # Layer 4 (Output): 128 -> 10
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass flattening the image and passing through the network."""
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.drop3(x)
        
        # Final layer (no activation or batchnorm, raw logits)
        x = self.fc4(x)
        return x

    def get_all_gates(self) -> torch.Tensor:
        """
        Collects and concatenates all gate tensors from all PrunableLinear layers.
        Useful for visualization (histogram of all gates).
        """
        all_gates = []
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().flatten())
        
        if not all_gates:
            return torch.tensor([])
            
        return torch.cat(all_gates)

    def get_model_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Returns the overall sparsity percentage of the entire model.
        """
        all_gates = self.get_all_gates()
        if all_gates.numel() == 0:
            return 0.0
            
        pruned_count = (all_gates < threshold).sum().item()
        total_count = all_gates.numel()
        return (pruned_count / total_count) * 100.0

    def get_layer_sparsity(self, threshold: float = 1e-2) -> Dict[str, float]:
        """
        Returns the sparsity percentage per PrunableLinear layer.
        """
        layer_sparsity = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                layer_sparsity[name] = module.get_sparsity(threshold)
        return layer_sparsity
