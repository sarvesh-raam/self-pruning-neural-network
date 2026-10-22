"""
Gradio demo for the Self-Pruning Neural Network.

Classifies an image with every exported model side by side, so you can see how
much accuracy survives as more of the network is pruned away. Runs on CPU and
only needs the compact sparse models in `models/` (see src/export.py).

    python app.py            # http://127.0.0.1:7860
"""
import glob
import json
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.export import load_compact

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)

torch.set_grad_enabled(False)


def load_models() -> list[tuple[torch.nn.Module, dict]]:
    models = [load_compact(p) for p in glob.glob(os.path.join(ROOT, "models", "lambda_*.pt"))]
    return sorted(models, key=lambda m: m[1]["lambda"])


MODELS = load_models()


def describe(metrics: dict) -> str:
    return (f"λ = {metrics['lambda']:g} · {metrics['sparsity']:.1f}% pruned · "
            f"{metrics['accuracy']:.1f}% test accuracy")


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return ((x - MEAN) / STD).unsqueeze(0)


def classify(image: Image.Image | None):
    if image is None:
        return [None] * len(MODELS)
    x = preprocess(image)
    outputs = []
    for model, _ in MODELS:
        probs = torch.softmax(model(x), dim=1)[0]
        outputs.append({c: float(p) for c, p in zip(CLASSES, probs)})
    return outputs


def results_table() -> list[list]:
    rows = []
    for _, m in MODELS:
        rows.append([f"{m['lambda']:g}", f"{m['accuracy']:.2f}%", f"{m['sparsity']:.2f}%",
                     f"{m['weights_remaining']:,} / {m['weights_total']:,}", f"{m['compression']:.1f}x"])
    return rows


def best_lambda_tag() -> str:
    path = os.path.join(ROOT, "results", "results.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            best = max(json.load(f), key=lambda r: r["accuracy"])
        return f"lambda_{best['lambda']:g}"
    return f"lambda_{MODELS[0][1]['lambda']:g}"


def figure(name: str) -> str | None:
    path = os.path.join(ROOT, "results", name)
    return path if os.path.exists(path) else None


EXPLAINER = r"""
Each weight $w_{ij}$ in a `PrunableLinear` layer is multiplied by a learnable gate
$g_{ij} = \sigma(s_{ij}) \in (0, 1)$. Training minimises

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \sum_{ij} \sigma(s_{ij})$$

The L1 term pulls every gate towards 0 with a steady force, whereas an L2 penalty's pull
fades as a value gets small. Only connections whose contribution to the classification
loss outweighs that pull stay open. After training, every gate below 0.01 is set to
exactly zero (hard pruning), and the accuracies shown here are measured on that truly
sparse network.

Higher λ → more connections removed → smaller model, lower accuracy.
"""

with gr.Blocks(title="Self-Pruning Neural Network", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Self-Pruning Neural Network\nA CIFAR-10 classifier that learned to delete most of its own weights "
                "during training. Upload an image (or pick an example) to compare the pruned models.")
    if not MODELS:
        gr.Markdown("**No models found in `models/`.** Run `python main.py --run_all` first.")
    else:
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="pil", label="Input image (resized to 32x32)", height=280)
                    button = gr.Button("Classify", variant="primary")
                    examples = sorted(glob.glob(os.path.join(ROOT, "assets", "examples", "*.png")))
                    if examples:
                        gr.Examples(examples, inputs=image, label="CIFAR-10 test images")
                with gr.Column(scale=2):
                    labels = [gr.Label(num_top_classes=3, label=describe(m)) for _, m in MODELS]
            button.click(classify, image, labels)
            image.change(classify, image, labels)

        with gr.Tab("Results"):
            gr.Dataframe(results_table(), headers=["λ", "Test accuracy", "Sparsity", "Weights kept", "Compression"],
                         interactive=False)
            with gr.Row():
                gr.Image(figure("lambda_comparison.png"), label="Trade-off", show_download_button=False)
                gr.Image(figure(f"gates_dist_{best_lambda_tag()}.png"), label="Gate distribution (best model)",
                         show_download_button=False)
            gr.Image(figure("layer_sparsity.png"), label="Sparsity per layer", show_download_button=False)

        with gr.Tab("How it works"):
            gr.Markdown(EXPLAINER, latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                     {"left": "$", "right": "$", "display": False}])
            gr.Markdown("Source: [github.com/sarvesh-raam/self-pruning-neural-network]"
                        "(https://github.com/sarvesh-raam/self-pruning-neural-network)")

if __name__ == "__main__":
    demo.launch()
