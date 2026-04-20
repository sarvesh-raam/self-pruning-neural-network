import argparse
import torch
import sys
import os
from rich.console import Console
from rich.table import Table

from src.model import SelfPruningNet
from src.train import Trainer
from src.visualize import plot_training_curves, plot_gate_distribution
from experiments.run_all import run_experiments

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network Training CLI")
    parser.add_argument('--lambda_val', type=float, default=0.001, help="L1 sparsity penalty coefficient (default: 0.001)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on (cuda or cpu)")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument('--run_all', action='store_true', help="Flag to run all 3 lambda experiments (ignores --lambda_val)")
    
    args = parser.parse_args()
    
    console.rule("[bold cyan]Self-Pruning Neural Network - Tredence Case Study[/bold cyan]")
    
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Epochs", str(args.epochs))
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Device", args.device)
    if args.run_all:
        table.add_row("Mode", "Run All Experiments")
    else:
        table.add_row("Lambda (λ)", str(args.lambda_val))
        
    console.print(table)
    print("\n")
    
    if args.run_all:
        console.print("[bold yellow]Initiating full experiment suite (λ = 0.0001, 0.001, 0.01)...[/bold yellow]")
        run_experiments(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    else:
        # Run a single experiment
        torch.manual_seed(42)
        if torch.cuda.is_available() and 'cuda' in args.device:
            torch.cuda.manual_seed_all(42)
            
        model = SelfPruningNet()
        trainer = Trainer(model, lambda_val=args.lambda_val, device=args.device)
        
        console.print(f"[bold green]Loading CIFAR-10 data...[/bold green]")
        trainer.load_cifar10(batch_size=args.batch_size)
        
        console.print(f"[bold green]Starting Training Loop...[/bold green]")
        history = trainer.train(epochs=args.epochs)
        
        console.print(f"[bold green]Evaluating final model...[/bold green]")
        acc, spars = trainer.evaluate()
        
        console.print(f"\n[bold white on green] Final Results [/bold white on green]")
        console.print(f"Accuracy: [bold]{acc:.2f}%[/bold]")
        console.print(f"Sparsity: [bold]{spars:.2f}%[/bold]")
        
        os.makedirs('results', exist_ok=True)
        ckpt_path = f'results/model_single.pth'
        trainer.save_checkpoint(ckpt_path)
        
        plot_training_curves(history, args.lambda_val, 'results/single_curves.png')
        plot_gate_distribution(trainer.model, args.lambda_val, 'results/single_gates.png')
        
        console.print(f"[bold cyan]Plots saved to results/ folder.[/bold cyan]")

if __name__ == '__main__':
    main()
