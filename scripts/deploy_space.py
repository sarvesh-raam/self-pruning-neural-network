"""
Publish the Gradio demo to a Hugging Face Space.

    pip install huggingface_hub
    hf auth login                     # or set HF_TOKEN
    python scripts/deploy_space.py --space <your-hf-username>/self-pruning-neural-network

Only what the demo needs is uploaded: app.py, the inference modules, the compact
sparse models, the result figures and a few example images. Training code and
the dataset stay out of the Space.
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SPACE_README = """---
title: Self-Pruning Neural Network
emoji: ✂️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: A CIFAR-10 network that learned to prune its own weights
---

Live demo for [self-pruning-neural-network](https://github.com/sarvesh-raam/self-pruning-neural-network).
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pillow
"""

FILES = ["app.py", "src/__init__.py", "src/model.py", "src/prunable_layer.py", "src/export.py", "LICENSE"]
DIRS = {"models": (".pt",), "assets/examples": (".png",), "results": (".png", ".json")}


def stage(target: str, gradio_version: str) -> None:
    for rel in FILES:
        os.makedirs(os.path.dirname(os.path.join(target, rel)) or target, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(target, rel))
    for rel, suffixes in DIRS.items():
        src = os.path.join(ROOT, rel)
        if not os.path.isdir(src):
            continue
        os.makedirs(os.path.join(target, rel), exist_ok=True)
        for name in os.listdir(src):
            if name.endswith(suffixes):
                shutil.copy2(os.path.join(src, name), os.path.join(target, rel, name))
    with open(os.path.join(target, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README.format(gradio_version=gradio_version))
    with open(os.path.join(target, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(SPACE_REQUIREMENTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    parser.add_argument("--gradio_version", default="5.50.0")
    args = parser.parse_args()

    if not any(f.endswith(".pt") for f in os.listdir(os.path.join(ROOT, "models"))):
        raise SystemExit("No models in models/. Run `python main.py --run_all` first.")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp, args.gradio_version)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy demo from GitHub",
                          # remove stale files from earlier deploys, but keep the Space's .gitattributes
                          delete_patterns=["*.py", "*.pt", "*.png", "*.json", "*.md", "*.txt", "LICENSE"])
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
