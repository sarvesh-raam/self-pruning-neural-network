# Sample Terminal Output

This file contains the output generated when running the training script on an NVIDIA RTX 5050.
Notice how the total loss drastically drops as the L1 Sparsity penalty forces the gates toward zero.

```text
D:\sar\self-pruning-neural-network>python main.py --run_all
────────────────────────────────── Self-Pruning Neural Network - Tredence Case Study ───────────────────────────────────
           Configuration
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter  ┃ Value               ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Epochs     │ 25                  │
│ Batch Size │ 256                 │
│ Device     │ cuda                │
│ Mode       │ Run All Experiments │
└────────────┴─────────────────────┘

Initiating full experiment suite (λ = 0.0001, 0.001, 0.01)...

============================================================
Running experiment for lambda = 0.0001
============================================================
Starting training for 25 epochs with lambda = 0.0001 on cuda
Epoch 01/25 | Total Loss: 166.6516 | Class Loss: 1.9106 | Acc: 41.13% | Sparsity: 0.00%
Epoch 05/25 | Total Loss: 153.9943 | Class Loss: 1.5941 | Acc: 48.61% | Sparsity: 0.00%
Epoch 10/25 | Total Loss: 112.6173 | Class Loss: 1.5114 | Acc: 50.55% | Sparsity: 0.00%
Epoch 15/25 | Total Loss: 62.9996 | Class Loss: 1.4669 | Acc: 53.24% | Sparsity: 0.00%
Epoch 20/25 | Total Loss: 34.8938 | Class Loss: 1.4231 | Acc: 54.86% | Sparsity: 0.00%
Epoch 25/25 | Total Loss: 21.1941 | Class Loss: 1.3955 | Acc: 55.65% | Sparsity: 0.00%
Checkpoint saved to results/model_lambda_0.0001.pth

============================================================
Running experiment for lambda = 0.01
============================================================
Starting training for 25 epochs with lambda = 0.01 on cuda
Epoch 01/25 | Total Loss: 16474.2988 | Class Loss: 1.9098 | Acc: 41.12% | Sparsity: 0.00%
Epoch 05/25 | Total Loss: 15213.6340 | Class Loss: 1.5934 | Acc: 47.68% | Sparsity: 0.00%
Epoch 10/25 | Total Loss: 10975.1964 | Class Loss: 1.5100 | Acc: 50.59% | Sparsity: 0.00%
Epoch 15/25 | Total Loss: 5932.8172 | Class Loss: 1.4688 | Acc: 52.69% | Sparsity: 0.00%
Epoch 20/25 | Total Loss: 3119.2174 | Class Loss: 1.4310 | Acc: 54.66% | Sparsity: 0.00%
Epoch 25/25 | Total Loss: 1759.3421 | Class Loss: 1.4071 | Acc: 55.31% | Sparsity: 0.00%
Checkpoint saved to results/model_lambda_0.01.pth

✅ All experiments finished! Results saved to 'results/' directory.
```
