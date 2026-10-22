import argparse
import sys

from rich.console import Console
from rich.table import Table

from experiments.run_all import DEFAULT_LAMBDAS, run_experiments
from src.train import TrainConfig, resolve_device

console = Console()


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None,
                        help=f"L1 gate penalty values to compare (default with --run_all: {' '.join(map(str, DEFAULT_LAMBDAS))})")
    parser.add_argument("--lambda_val", type=float, default=defaults.lambda_val, help="Single lambda to train")
    parser.add_argument("--run_all", action="store_true", help="Train every default lambda and compare them")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate for weights")
    parser.add_argument("--gate_lr", type=float, default=defaults.gate_lr, help="Learning rate for gate scores")
    parser.add_argument("--warmup_epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="Gate value counted as pruned")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models", help="Where compact sparse models are exported")
    return parser.parse_args()


def main() -> None:
    # Windows consoles default to cp1252, which cannot print symbols like λ when output is redirected
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    lambdas = args.lambdas or (DEFAULT_LAMBDAS if args.run_all else [args.lambda_val])
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, gate_lr=args.gate_lr,
                         warmup_epochs=args.warmup_epochs, threshold=args.threshold, seed=args.seed,
                         device=resolve_device(args.device))

    console.rule("[bold cyan]Self-Pruning Neural Network[/bold cyan]")
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    for name, value in [("Lambdas", ", ".join(f"{l:g}" for l in lambdas)), ("Epochs", config.epochs),
                        ("Batch size", config.batch_size), ("LR (weights / gates)", f"{config.lr:g} / {config.gate_lr:g}"),
                        ("Prune threshold", config.threshold), ("Device", config.device)]:
        table.add_row(name, str(value))
    console.print(table)

    run_experiments(lambdas, config, out_dir=args.out_dir, models_dir=args.models_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
