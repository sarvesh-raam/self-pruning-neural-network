import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List
from .model import SelfPruningNet

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_gate_distribution(model: SelfPruningNet, lambda_val: float, save_path: str) -> None:
    """Plots a histogram of all gate values in the network."""
    all_gates = model.get_all_gates().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(all_gates, bins=50, kde=False, color='steelblue')
    
    plt.title(f'Gate Value Distribution (λ = {lambda_val})', fontsize=18, pad=20)
    plt.xlabel('Gate Value (Sigmoid Output)', fontsize=14)
    plt.ylabel('Count (Number of Weights)', fontsize=14)
    
    # Annotate the "pruned" spike at 0
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7)
    plt.text(0.02, plt.ylim()[1]*0.9, 'Pruned (Gates ≈ 0)', color='red', fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(history: Dict[str, List[float]], lambda_val: float, save_path: str) -> None:
    """Plots training curves: losses, accuracy, and sparsity over epochs."""
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Losses
    ax1.plot(epochs, history['class_loss'], label='Classification Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['sparsity_loss'], label='Sparsity Loss', color='green', linewidth=2)
    ax1.set_title(f'Training Losses (λ = {lambda_val})', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend()
    
    # Plot 2: Accuracy
    ax2.plot(epochs, history['accuracy'], label='Test Accuracy', color='purple', linewidth=2)
    ax2.set_title('Test Accuracy over Time', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_ylim([0, 100])
    
    # Plot 3: Sparsity
    ax3.plot(epochs, history['sparsity'], label='Network Sparsity', color='orange', linewidth=2)
    ax3.set_title('Network Sparsity over Time', fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=14)
    ax3.set_ylabel('Sparsity (%)', fontsize=14)
    ax3.set_ylim([0, 100])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_lambda_comparison(all_results: List[Dict], save_path: str) -> None:
    """Plots a bar chart comparing accuracy vs sparsity across lambda values."""
    lambdas = [str(r['lambda']) for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    sparsities = [r['sparsity'] for r in all_results]
    
    x = np.arange(len(lambdas))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'purple'
    ax1.set_xlabel('Lambda (λ) Penalty', fontsize=14)
    ax1.set_ylabel('Test Accuracy (%)', color=color1, fontsize=14)
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 100])
    
    ax2 = ax1.twinx()
    color2 = 'orange'
    ax2.set_ylabel('Sparsity (%)', color=color2, fontsize=14)
    bars2 = ax2.bar(x + width/2, sparsities, width, label='Sparsity', color=color2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 100])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas)
    plt.title('Sparsity vs Accuracy Trade-off across Lambdas', fontsize=18, pad=20)
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_sparsity_heatmap(model: SelfPruningNet, save_path: str) -> None:
    """Plots a heatmap showing the sparsity level of each PrunableLinear layer."""
    layer_sparsity = model.get_layer_sparsity()
    
    names = list(layer_sparsity.keys())
    values = np.array(list(layer_sparsity.values())).reshape(1, -1)
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(values, annot=True, cmap='YlOrRd', fmt='.1f',
                xticklabels=names, yticklabels=['Sparsity %'], cbar=False)
    
    plt.title('Sparsity Distribution Across Layers', fontsize=16, pad=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
