"""
Export the compact models in models/ for the in-browser demo in web/.

Each model becomes one little-endian binary file. Per layer it holds the
flat indices of the kept weights (int32), their values (float32) and the
bias (float32). web/models/manifest.json lists the models, their metrics and
where each array starts. Result figures are copied next to the page.

    python scripts/export_web.py
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
WEB = os.path.join(ROOT, "web")
FIGURES = ["lambda_comparison.png", "layer_sparsity.png"]


def export_model(path: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    metrics = blob["metrics"]
    tag = os.path.splitext(os.path.basename(path))[0]
    layers, chunks, offset = [], [], 0
    for name, data in blob["layers"].items():
        arrays = {
            "index": data["index"].numpy().astype("<i4"),
            "value": data["value"].float().numpy().astype("<f4"),
            "bias": data["bias"].float().numpy().astype("<f4"),
        }
        entry = {"name": name, "shape": list(data["shape"]), "nnz": int(arrays["index"].size)}
        for key, arr in arrays.items():
            entry[f"{key}_offset"] = offset
            chunks.append(arr.tobytes())
            offset += arr.nbytes
        layers.append(entry)
    with open(os.path.join(WEB, "models", f"{tag}.bin"), "wb") as f:
        f.write(b"".join(chunks))
    keep = ("lambda", "accuracy", "accuracy_soft", "sparsity", "weights_total", "weights_remaining", "compression")
    return {"file": f"models/{tag}.bin", "bytes": offset, "layers": layers, **{k: metrics[k] for k in keep}}


def main() -> None:
    os.makedirs(os.path.join(WEB, "models"), exist_ok=True)
    paths = glob.glob(os.path.join(ROOT, "models", "lambda_*.pt"))
    if not paths:
        raise SystemExit("No models in models/. Run `python main.py --run_all` first.")
    models = sorted((export_model(p) for p in paths), key=lambda m: m["lambda"])
    with open(os.path.join(WEB, "models", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, indent=1)

    os.makedirs(os.path.join(WEB, "img", "examples"), exist_ok=True)
    for name in FIGURES:
        shutil.copy2(os.path.join(ROOT, "results", name), os.path.join(WEB, "img", name))
    best = max(models, key=lambda m: m["sparsity"] if m["accuracy"] > 50 else -1)
    shutil.copy2(os.path.join(ROOT, "results", f"gates_dist_lambda_{best['lambda']:g}.png"),
                 os.path.join(WEB, "img", "gates_dist.png"))
    for p in glob.glob(os.path.join(ROOT, "assets", "examples", "*.png")):
        shutil.copy2(p, os.path.join(WEB, "img", "examples", os.path.basename(p)))
    for m in models:
        print(f"{m['file']}: {m['bytes'] / 1e6:.2f} MB, {m['sparsity']:.1f}% pruned")


if __name__ == "__main__":
    main()
