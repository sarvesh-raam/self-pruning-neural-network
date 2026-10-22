"""
Publish the demo to a Hugging Face Space.

    pip install huggingface_hub
    python -c "from huggingface_hub import login; login()"     # or set HF_TOKEN
    python scripts/deploy_space.py --space <your-hf-username>/self-pruning-neural-network

By default this deploys the static site in web/ (the model runs in the visitor's
browser), which is free on every Hugging Face account. `--gradio` deploys the
Python app.py instead, which Hugging Face only hosts for PRO accounts.
Training code and the dataset never leave your machine.
"""
import argparse
import os
import shutil
import subprocess
import sys
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

STATIC_README = """---
title: Self-Pruning Neural Network
emoji: ✂️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: A CIFAR-10 network that pruned 98% of its own weights
---
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pillow
"""

FILES = ["app.py", "src/__init__.py", "src/model.py", "src/prunable_layer.py", "src/export.py", "LICENSE"]
DIRS = {"models": (".pt",), "assets/examples": (".png",), "results": (".png", ".json")}


def stage_static(target: str) -> None:
    # Re-export so the site always matches the models in models/
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_web.py")], check=True)
    shutil.copytree(os.path.join(ROOT, "web"), target, dirs_exist_ok=True)
    with open(os.path.join(target, "README.md"), "w", encoding="utf-8") as f:
        f.write(STATIC_README)


def stage_gradio(target: str, gradio_version: str) -> None:
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
    parser.add_argument("--gradio", action="store_true", help="Deploy the Gradio app (needs Hugging Face PRO)")
    parser.add_argument("--gradio_version", default="5.50.0")
    args = parser.parse_args()

    if not any(f.endswith(".pt") for f in os.listdir(os.path.join(ROOT, "models"))):
        raise SystemExit("No models in models/. Run `python main.py --run_all` first.")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.space, repo_type="space", space_sdk="gradio" if args.gradio else "static", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if args.gradio:
            stage_gradio(tmp, args.gradio_version)
        else:
            stage_static(tmp)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy demo from GitHub",
                          # remove stale files from earlier deploys, but keep the Space's .gitattributes
                          delete_patterns=["*.py", "*.pt", "*.bin", "*.png", "*.json", "*.md", "*.txt", "*.html", "*.js", "LICENSE"])
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
