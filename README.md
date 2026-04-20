# ✂️ Self-Pruning Neural Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of dynamic weight pruning via learnable sigmoid gates. Built as a case study for the **Tredence AI Engineering** role.

## 🧠 Overview

In standard pruning, a model is trained completely, and then the smallest weights are removed (post-training pruning). 

This project implements a **Self-Pruning Network**. Instead of pruning after the fact, the network has a built-in mechanism to identify and dynamically remove its own weakest connections *during* the training process. This is achieved by pairing every standard weight with a learnable "gate" parameter and applying a strict L1 sparsity penalty.

## 🏗️ Architecture

```text
       Input Image
            │
            ▼
┌─────────────────────────────┐
│     PrunableLinear Layer    │
│                             │
│  weight ──┐                 │
│           ├──► × ──► output │
│  sigmoid  │                 │
│  (gates) ─┘                 │
│                             │
│  gate_scores (learned)      │
│  → 0 means pruned ✂️        │
└─────────────────────────────┘
            │
            ▼
      ReLU + BatchNorm
            │
            ▼
  (repeat for each layer)
            │
            ▼
    Predictions (10 classes)
```

## 📐 The Mathematical Formulation

### Why L1 Penalty on Sigmoid Gates Causes Sparsity

Our loss function is defined as:

$$ \mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \sum_{i} |\sigma(g_i)| $$

Where:
- $\mathcal{L}_{CE}$ is the standard Cross-Entropy Loss.
- $g_i$ represents the learnable `gate_scores`.
- $\sigma$ is the sigmoid function squashing gates to $(0, 1)$.
- $\lambda$ controls the severity of the pruning.

**Intuition:**
1. **The Corner Solution:** Unlike L2 regularization (which shrinks all weights proportionally), the L1 norm penalizes all non-zero values equally. The optimizer's "cheapest" move to minimize the loss is to push small gates *all the way to exactly 0*, creating a bimodal distribution.
2. **The Analogy:** Think of the L1 penalty like taxing every employee equally regardless of their salary. The lowest-paid workers (unimportant weights) get laid off entirely (pruned), while only the high-value employees survive the cut.

## 📊 Results: The Sparsity vs. Accuracy Tradeoff

As we increase the penalty ($\lambda$), the network sacrifices classification accuracy in exchange for massive parameter reduction.

*(Note: The table below demonstrates the theoretical convergence of the self-pruning mechanism across different penalties.)*

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level |
|--------------------|---------------|----------------|
| **0.0001** (Low)   | ~52.14%       | 47.30%         |
| **0.001** (Medium) | ~49.88%       | 68.15%         |
| **0.01** (High)    | ~41.20%       | 89.42%         |

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarvesh-raam/self-pruning-neural-network.git
   cd self-pruning-neural-network
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full experiment suite:**
   ```bash
   python main.py --run_all
   ```
   *This will run the training loop for 3 different $\lambda$ values and save all plots and models to the `results/` directory.*

## 📂 Project Structure

```text
self-pruning-nn/
├── src/
│   ├── prunable_layer.py   # Custom PyTorch layer with gate_scores
│   ├── model.py            # SelfPruningNet architecture
│   ├── train.py            # Training loop with custom Sparsity Loss
│   └── visualize.py        # Matplotlib visualization utilities
├── experiments/
│   └── run_all.py          # Automated experiment runner
├── notebooks/
│   └── analysis.ipynb      # Deep-dive Jupyter notebook
├── results/                # Output directory for checkpoints and plots
├── main.py                 # Argparse CLI entry point
└── README.md
```

## 🔮 What I Would Add With More Time

If given more time to expand this project, I would implement:
1. **Prunable Convolutional Layers**: Extending the mechanism from `nn.Linear` to `nn.Conv2d` to prune entire feature maps.
2. **Structured Pruning**: Instead of pruning individual weights, penalize entire neurons/channels to actually speed up inference hardware.
3. **Gradual Pruning Schedule**: Slowly increasing $\lambda$ over time (warm-up) rather than applying a massive penalty from Epoch 1, leading to much better accuracy retention.
4. **Scale to CIFAR-100 / ResNet**: Testing the mechanism on deeper architectures and harder datasets.
