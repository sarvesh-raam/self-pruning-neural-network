import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: works on servers, CI and Spaces
import matplotlib.pyplot as plt
import numpy as np

from .model import SelfPruningNet

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
})
ACCENT, SECOND, PRUNED = "#2563eb", "#f97316", "#dc2626"


def _save(fig: plt.Figure, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_gate_distribution(model: SelfPruningNet, lambda_val: float, save_path: str,
                           threshold: float = 1e-2) -> None:
    """
    Histogram of every gate value in the network.

    The y-axis is logarithmic: a successful run has a spike of hundreds of
    thousands of gates at 0 and a much smaller cluster of surviving gates, and on
    a linear axis the survivors would be invisible.
    """
    gates = model.get_all_gates().cpu().numpy()
    pruned = (gates < threshold).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gates, bins=100, range=(0, 1), color=ACCENT, log=True)
    ax.axvline(threshold, color=PRUNED, linestyle="--", linewidth=1.2)
    ax.text(threshold + 0.015, 0.95, f"threshold {threshold:g}\n{pruned:.1f}% pruned",
            color=PRUNED, transform=ax.get_xaxis_transform(), va="top")
    ax.set_title(f"Final gate values (λ = {lambda_val:g})")
    ax.set_xlabel("gate value  sigmoid(gate_score)")
    ax.set_ylabel("number of weights (log scale)")
    _save(fig, save_path)


def plot_training_curves(history: Dict[str, List[float]], lambda_val: float, save_path: str) -> None:
    """Classification loss, test accuracy and sparsity per epoch."""
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, history["class_loss"], color=ACCENT, linewidth=2)
    axes[0].set_title("Cross-entropy loss (train)")
    axes[1].plot(epochs, history["accuracy"], color=ACCENT, linewidth=2)
    axes[1].set_title("Test accuracy (%)")
    axes[2].plot(epochs, history["sparsity"], color=SECOND, linewidth=2)
    axes[2].set_title("Sparsity (% of gates < threshold)")
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.suptitle(f"Training dynamics, λ = {lambda_val:g}", y=1.02)
    _save(fig, save_path)


def plot_lambda_comparison(all_results: List[Dict], save_path: str) -> None:
    """Accuracy vs sparsity for each lambda: the trade-off curve."""
    sparsity = [r["sparsity"] for r in all_results]
    accuracy = [r["accuracy"] for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sparsity, accuracy, color=ACCENT, linewidth=1.5, marker="o", markersize=9)
    for r in all_results:
        ax.annotate(f"λ = {r['lambda']:g}\n{r['compression']:.0f}x smaller",
                    (r["sparsity"], r["accuracy"]), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("sparsity (% of weights pruned)")
    ax.set_ylabel("test accuracy after hard pruning (%)")
    ax.set_title("Sparsity vs accuracy trade-off")
    ax.set_xlim(min(sparsity) - 5, 100)
    ax.set_ylim(min(accuracy) - 5, max(accuracy) + 5)
    _save(fig, save_path)


def plot_layer_sparsity_heatmap(all_results: List[Dict], save_path: str) -> None:
    """Rows = lambda, columns = layer, cell = % of that layer's weights pruned."""
    layers = list(all_results[0]["layer_sparsity"].keys())
    values = np.array([[r["layer_sparsity"][name] for name in layers] for r in all_results])

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.7 * len(all_results)))
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(layers)), labels=layers)
    ax.set_yticks(range(len(all_results)), labels=[f"λ = {r['lambda']:g}" for r in all_results])
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center",
                    color="white" if values[i, j] > 60 else "black")
    fig.colorbar(im, ax=ax, label="% pruned")
    ax.set_title("Sparsity per layer")
    _save(fig, save_path)
