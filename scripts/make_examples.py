"""Save one CIFAR-10 test image per class (upscaled for display) as demo examples in assets/examples/."""
import os
import sys

import torchvision
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import CIFAR_CLASSES  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")


def main(data_dir: str = "./data") -> None:
    os.makedirs(OUT, exist_ok=True)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    seen = set()
    for image, label in zip(test.data, test.targets):
        if label in seen:
            continue
        seen.add(label)
        Image.fromarray(image).resize((128, 128), Image.NEAREST).save(
            os.path.join(OUT, f"{label}_{CIFAR_CLASSES[label]}.png"))
        if len(seen) == len(CIFAR_CLASSES):
            break


if __name__ == "__main__":
    main()
