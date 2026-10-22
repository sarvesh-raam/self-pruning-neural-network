"""
Compact export of a hard-pruned SelfPruningNet.

Training checkpoints hold weight + gate_scores + mask in float32 for every
connection. For inference only the surviving effective weights
(weight * gate) are needed, so they are stored sparsely: the flat index of each
kept weight (int32) and its value (float16). At 90%+ sparsity the file is a
small fraction of the dense model, which is what makes the demo cheap to host.
"""
import os
from typing import Dict

import torch

from .model import SelfPruningNet

# sigmoid(30) == 1.0 in float32, so an imported gate passes the weight through unchanged
_OPEN_GATE = 30.0


@torch.no_grad()
def export_compact(model: SelfPruningNet, path: str, metrics: Dict | None = None) -> int:
    """Write the sparse model to `path`. Returns the file size in bytes."""
    layers = {}
    for name, layer in model.prunable_layers():
        effective = (layer.weight * layer.gates()).flatten().cpu()
        keep = torch.nonzero(layer.mask.flatten().cpu(), as_tuple=True)[0]
        layers[name] = {
            "shape": tuple(layer.weight.shape),
            "index": keep.to(torch.int32),
            "value": effective[keep].to(torch.float16),
            "bias": layer.bias.detach().cpu().to(torch.float16),
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_config": model.config, "layers": layers, "metrics": metrics or {}}, path)
    return os.path.getsize(path)


@torch.no_grad()
def load_compact(path: str, device: str = "cpu") -> tuple[SelfPruningNet, Dict]:
    """Rebuild an inference-ready SelfPruningNet from `export_compact` output."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = SelfPruningNet(**blob["model_config"])
    modules = dict(model.prunable_layers())
    for name, data in blob["layers"].items():
        layer = modules[name]
        weight = torch.zeros(layer.weight.numel())
        mask = torch.zeros(layer.weight.numel())
        idx = data["index"].long()
        weight[idx] = data["value"].float()
        mask[idx] = 1.0
        layer.weight.copy_(weight.view(data["shape"]))
        layer.mask.copy_(mask.view(data["shape"]))
        layer.gate_scores.fill_(_OPEN_GATE)
        layer.bias.copy_(data["bias"].float())
    return model.to(device).eval(), blob["metrics"]
