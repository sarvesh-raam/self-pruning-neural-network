import os
import torch
import sys

# Ensure we can import from src when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SelfPruningNet
from src.train import Trainer
from src.visualize import (
    plot_gate_distribution, plot_training_curves,
    plot_lambda_comparison, plot_layer_sparsity_heatmap
)

def run_experiments(epochs: int = 25, batch_size: int = 256, device: str = 'cuda') -> None:
    lambdas = [0.0001, 0.001, 0.01]
    all_results = []
    
    os.makedirs('results', exist_ok=True)
    
    for l_val in lambdas:
        print(f"\n{'='*60}\nRunning experiment for lambda = {l_val}\n{'='*60}")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        model = SelfPruningNet()
        trainer = Trainer(model, lambda_val=l_val, device=device)
        trainer.load_cifar10(batch_size=batch_size)
        
        history = trainer.train(epochs=epochs)
        final_acc, final_spars = trainer.evaluate()
        
        # Save checkpoints and generate per-lambda plots
        ckpt_path = f'results/model_lambda_{l_val}.pth'
        trainer.save_checkpoint(ckpt_path)
        
        plot_gate_distribution(trainer.model, l_val, f'results/gates_dist_lambda_{l_val}.png')
        plot_training_curves(history, l_val, f'results/curves_lambda_{l_val}.png')
        plot_layer_sparsity_heatmap(trainer.model, f'results/heatmap_lambda_{l_val}.png')
        
        all_results.append({
            'lambda': l_val,
            'accuracy': final_acc,
            'sparsity': final_spars
        })
        
    # Generate final comparison plot
    plot_lambda_comparison(all_results, 'results/lambda_comparison.png')
    
    # Save summary table
    summary_path = 'results/summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("### Experiment Results\n\n")
        f.write("| Lambda | Test Accuracy | Sparsity Level |\n")
        f.write("|--------|---------------|----------------|\n")
        for res in all_results:
            f.write(f"| {res['lambda']} | {res['accuracy']:.2f}% | {res['sparsity']:.2f}% |\n")
            
    print(f"\n✅ All experiments finished! Results saved to 'results/' directory.")
    print("\nSummary Table:")
    with open(summary_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
if __name__ == '__main__':
    device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_experiments(epochs=25, batch_size=256, device=device_to_use)
