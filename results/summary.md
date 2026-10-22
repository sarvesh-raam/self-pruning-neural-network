| Lambda | Test Accuracy | Sparsity Level (%) | Weights kept | Compact model size |
|--------|---------------|--------------------|--------------|--------------------|
| 1e-06 | 57.42% | 84.52% | 269,027 / 1,737,984 | 1.62 MB |
| 1e-05 | 53.10% | 98.11% | 32,795 / 1,737,984 | 0.20 MB |
| 5e-05 | 47.11% | 99.56% | 7,674 / 1,737,984 | 0.05 MB |

Best accuracy: λ = 1e-06. Accuracy is measured after hard pruning (every gate below the threshold set to exactly 0).
