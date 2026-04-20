# Contributing to Self-Pruning Neural Network

First off, thank you for considering contributing to this repository! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## How Can I Contribute?

### Reporting Bugs
If you find a bug in the code, please open an issue. Provide as much context as possible:
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- OS, Python version, and PyTorch version

### Suggesting Enhancements
Have an idea for a cool new feature? (e.g., Structured Pruning, Gradual Pruning Schedules, adapting to ResNet).
- Open an issue detailing your feature proposal.
- If you've already started working on it, feel free to open a Pull Request!

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes (via the GitHub Actions CI).
4. Make sure your code is clean and fully type-annotated.
5. Issue the pull request.

## Development Setup
```bash
git clone https://github.com/sarvesh-raam/self-pruning-neural-network.git
cd self-pruning-neural-network
pip install -r requirements.txt
python main.py --epochs 1 --batch_size 16 --device cpu
```

Thank you!
