import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.export import export_compact, load_compact
from src.model import SelfPruningNet
from src.prunable_layer import PrunableLinear
from src.train import TensorLoader, Trainer, TrainConfig, set_seed


@pytest.fixture(autouse=True)
def seed():
    set_seed(0)


def test_forward_matches_manual_computation():
    layer = PrunableLinear(8, 4)
    x = torch.randn(5, 8)
    expected = x @ (layer.weight * torch.sigmoid(layer.gate_scores)).T + layer.bias
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_gradients_flow_to_weights_and_gates():
    layer = PrunableLinear(8, 4)
    layer(torch.randn(5, 8)).sum().backward()
    for p in (layer.weight, layer.gate_scores, layer.bias):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_gate_scores_are_registered_parameters():
    layer = PrunableLinear(8, 4)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"weight", "bias", "gate_scores"}
    assert "mask" in dict(layer.named_buffers())


def test_sparsity_loss_is_sum_of_gates():
    model = SelfPruningNet(hidden=(16, 8))
    expected = sum(torch.sigmoid(m.gate_scores).sum() for _, m in model.prunable_layers())
    assert torch.allclose(model.sparsity_loss(), expected)
    model.sparsity_loss().backward()
    assert all(m.gate_scores.grad is not None for _, m in model.prunable_layers())


def test_hard_prune_removes_connections_exactly():
    layer = PrunableLinear(4, 2)
    with torch.no_grad():
        layer.gate_scores.fill_(5.0)
        layer.gate_scores[0, 1] = -10.0  # sigmoid ~ 4.5e-5, below 1e-2
    assert layer.get_sparsity(1e-2) == pytest.approx(100 / 8)
    assert layer.prune(1e-2) == 1

    # Input 1 should no longer influence output 0 at all
    x = torch.zeros(1, 4)
    base = layer(x)
    x[0, 1] = 1000.0
    assert torch.allclose(layer(x)[0, 0], base[0, 0])
    # and the pruned gate gets no gradient from the penalty any more
    layer.gates().sum().backward()
    assert layer.gate_scores.grad[0, 1] == 0


def test_compact_export_roundtrip(tmp_path):
    model = SelfPruningNet(hidden=(32, 16)).eval()
    with torch.no_grad():
        for _, m in model.prunable_layers():
            m.gate_scores.uniform_(-8, 4)
    model.prune(1e-2)
    x = torch.randn(3, 3, 32, 32)
    path = tmp_path / "m.pt"
    export_compact(model, str(path))
    restored, _ = load_compact(str(path))
    # float16 storage -> small tolerance
    assert torch.allclose(model(x), restored(x), atol=1e-2)
    assert restored.get_model_sparsity() == pytest.approx(model.get_model_sparsity())


def test_training_with_penalty_actually_prunes():
    """End-to-end on a tiny synthetic task: a strong lambda must produce real sparsity."""
    n = 512
    images = torch.randint(0, 256, (n, 32, 32, 3), dtype=torch.uint8)
    labels = (images[:, 0, 0, 0] > 127).long()  # depends on one pixel only
    train = TensorLoader(images, labels, 32, shuffle=True, augment=False, device="cpu")
    test = TensorLoader(images, labels, 256, shuffle=False, augment=False, device="cpu")

    # 16 steps x 30 epochs: enough Adam steps for a gate score to travel from +2 to well below -4.6
    config = TrainConfig(lambda_val=1e-3, epochs=30, warmup_epochs=1, gate_lr=0.1, device="cpu")
    trainer = Trainer(SelfPruningNet(hidden=(8,)), config)
    trainer.set_data(train, test)
    trainer.train(log=lambda *_: None)
    metrics = trainer.finalize()
    assert metrics["sparsity"] > 50
    assert metrics["weights_remaining"] < metrics["weights_total"]
