import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Sequence

# Allow `python experiments/run_all.py` as well as `python main.py --run_all`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.export import export_compact
from src.model import SelfPruningNet
from src.train import Trainer, TrainConfig, load_cifar10, set_seed
from src.visualize import (
    plot_gate_distribution, plot_lambda_comparison,
    plot_layer_sparsity_heatmap, plot_training_curves,
)

DEFAULT_LAMBDAS = (1e-6, 1e-5, 5e-5)


def run_experiment(config: TrainConfig, loaders, out_dir: str, models_dir: str) -> Dict:
    """Train one model, hard-prune it, save checkpoint + compact export + plots, return its metrics."""
    set_seed(config.seed)
    trainer = Trainer(SelfPruningNet(), config)
    trainer.set_data(*loaders)
    history = trainer.train()

    tag = f"lambda_{config.lambda_val:g}"
    # Plot the learned gates before hard pruning, so the figure shows what training produced
    plot_gate_distribution(trainer.model, config.lambda_val, f"{out_dir}/gates_dist_{tag}.png", config.threshold)
    plot_training_curves(history, config.lambda_val, f"{out_dir}/curves_{tag}.png")

    metrics = trainer.finalize()
    metrics["history"] = history
    trainer.save_checkpoint(f"{out_dir}/model_{tag}.pth", metrics)
    metrics["compact_bytes"] = export_compact(trainer.model, f"{models_dir}/{tag}.pt",
                                              {k: v for k, v in metrics.items() if k != "history"})
    print(f"lambda = {config.lambda_val:g}: accuracy {metrics['accuracy']:.2f}% "
          f"(before hard prune {metrics['accuracy_soft']:.2f}%), sparsity {metrics['sparsity']:.2f}%, "
          f"{metrics['compression']:.1f}x fewer weights")
    return metrics


def write_summary(results: List[Dict], path: str) -> None:
    best = max(results, key=lambda r: r["accuracy"])
    lines = [
        "| Lambda | Test Accuracy | Sparsity Level (%) | Weights kept | Compact model size |",
        "|--------|---------------|--------------------|--------------|--------------------|",
    ]
    for r in results:
        lines.append(f"| {r['lambda']:g} | {r['accuracy']:.2f}% | {r['sparsity']:.2f}% | "
                     f"{r['weights_remaining']:,} / {r['weights_total']:,} | {r['compact_bytes'] / 1e6:.2f} MB |")
    lines.append("")
    lines.append(f"Best accuracy: λ = {best['lambda']:g}. Accuracy is measured after hard pruning "
                 "(every gate below the threshold set to exactly 0).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_experiments(lambdas: Sequence[float] = DEFAULT_LAMBDAS, base_config: TrainConfig | None = None,
                    out_dir: str = "results", models_dir: str = "models", data_dir: str = "./data") -> List[Dict]:
    base_config = base_config or TrainConfig()
    os.makedirs(out_dir, exist_ok=True)
    loaders = load_cifar10(data_dir, base_config.batch_size, base_config.device)  # loaded once, reused

    results = []
    for lam in lambdas:
        print(f"\n{'=' * 60}\nExperiment: lambda = {lam:g}\n{'=' * 60}")
        results.append(run_experiment(replace(base_config, lambda_val=lam), loaders, out_dir, models_dir))

    plot_lambda_comparison(results, f"{out_dir}/lambda_comparison.png")
    plot_layer_sparsity_heatmap(results, f"{out_dir}/layer_sparsity.png")
    with open(f"{out_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_summary(results, f"{out_dir}/summary.md")

    with open(f"{out_dir}/summary.md", encoding="utf-8") as f:
        print("\nAll experiments finished.\n\n" + f.read())
    return results


if __name__ == "__main__":
    run_experiments()
