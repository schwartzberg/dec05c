"""Schwartzberg, Paul. CSCI E-89 HW5 Problem 1: reproduce the tiger transforms."""

# Import the tools used below and show the active Python environment.
from pathlib import Path
from urllib.request import Request, urlopen
from io import BytesIO
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ASSET_DIR = Path("assets")
OUTPUT_DIR = Path("outputs/problem1")
ASSET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__)


# Use the saved photograph. Download it only if the local copy is missing.
TIGER_PAGE = "https://commons.wikimedia.org/wiki/File:Panthera_tigris_tigris_original.jpg"
TIGER_URL = "https://upload.wikimedia.org/wikipedia/commons/f/f9/Panthera_tigris_tigris_original.jpg"
TIGER_PATH = ASSET_DIR / "tiger.jpg"
if not TIGER_PATH.exists():
    request = Request(TIGER_URL, headers={"User-Agent": "CSCI-E89-HW5/1.0 (educational image transformations)"})
    with urlopen(request, timeout=45) as response:
        downloaded = response.read()
    with Image.open(BytesIO(downloaded)) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((960, 960))  # Keep the aspect ratio; limit the longest side.
        picture.save(TIGER_PATH, quality=92)

with Image.open(TIGER_PATH) as picture:
    tiger = picture.convert("RGB")
print("Working image:", TIGER_PATH, "| width x height:", tiger.size)


class DiagonalStretch:
    """Stretch along the image's top-left to bottom-right diagonal about its centre."""

    def __init__(self, factor=1.10, fill=(255, 255, 255)):
        self.factor = float(factor)
        self.fill = fill
        if self.factor < 1:
            raise ValueError("This demonstration expects a stretch factor of at least one.")

    def __call__(self, image):
        width, height = image.size
        # A unit vector gives the diagonal's direction, including a non-square image.
        diagonal = np.array([width, height], dtype=float)
        diagonal /= np.linalg.norm(diagonal)
        # Add 10% only to the component parallel to that diagonal.
        forward = np.eye(2) + (self.factor - 1) * np.outer(diagonal, diagonal)
        # Positive entries let us find the bounding canvas from width and height.
        out_width, out_height = np.ceil(forward @ [width, height]).astype(int)
        original_centre = np.array([width, height], dtype=float) / 2
        output_centre = np.array([out_width, out_height], dtype=float) / 2
        # Pillow needs the reverse mapping: for each output pixel, where to sample.
        inverse = np.linalg.inv(forward)
        offset = original_centre - inverse @ output_centre
        coefficients = (
            inverse[0, 0], inverse[0, 1], offset[0],
            inverse[1, 0], inverse[1, 1], offset[1],
        )
        return image.transform(
            (int(out_width), int(out_height)), Image.Transform.AFFINE,
            coefficients, resample=Image.Resampling.BILINEAR, fillcolor=self.fill,
        )


# Each Compose starts from the same photograph and ends with tensor conversion.
# p=1 forces the flips to happen; equal rotation bounds fix the angle.
tiger_transforms = {
    "horizontal_axis_flip": transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),  # Reflection about a horizontal axis.
        transforms.ToTensor(),
    ]),
    "vertical_axis_flip": transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Reflection about a vertical axis.
        transforms.ToTensor(),
    ]),
    "diagonal_stretch_10_percent": transforms.Compose([
        DiagonalStretch(factor=1.10),
        transforms.ToTensor(),
    ]),
    "rotate_left_10_degrees": transforms.Compose([
        transforms.RandomRotation((10, 10), interpolation=InterpolationMode.BILINEAR,
                                  expand=True, fill=255),
        transforms.ToTensor(),
    ]),
    "rotate_right_15_degrees": transforms.Compose([
        transforms.RandomRotation((-15, -15), interpolation=InterpolationMode.BILINEAR,
                                  expand=True, fill=255),
        transforms.ToTensor(),
    ]),
}
# Each tensor has shape (three colour channels, height, width), with values in [0, 1].
tiger_results = {name: pipeline(tiger) for name, pipeline in tiger_transforms.items()}


# Check the mathematical meaning of the stretch independently of the photograph.
u = np.array(tiger.size, dtype=float)
u /= np.linalg.norm(u)
v = np.array([-u[1], u[0]])  # A unit direction perpendicular to the chosen diagonal.
A = np.eye(2) + 0.10 * np.outer(u, u)
print(f"Length scale along the diagonal: {np.linalg.norm(A @ u):.2f}")
print(f"Length scale across the diagonal: {np.linalg.norm(A @ v):.2f}")
print("These are the geometric scales before resampling pixels.")


# Display the original and five separate results; save a static report and images.
labels = {
    "horizontal_axis_flip": "Horizontal-axis reflection\nTop and bottom swap",
    "vertical_axis_flip": "Vertical-axis reflection\nLeft and right swap",
    "diagonal_stretch_10_percent": "10% stretch along the diagonal\nTop left to bottom right",
    "rotate_left_10_degrees": "10 degrees left\nCounter-clockwise",
    "rotate_right_15_degrees": "15 degrees right\nClockwise",
}
shown = {"original": transforms.ToTensor()(tiger), **tiger_results}
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
manifest = {"author": "Schwartzberg, Paul", "source_page": TIGER_PAGE,
            "source_url": TIGER_URL, "image_credit": "John and Karen Hollingsworth, U.S. Fish and Wildlife Service; public domain in the United States",
            "stretch_factor": 1.10, "results": {}}
for ax, (name, tensor) in zip(axes.flat, shown.items()):
    # Matplotlib expects height, width, channels; tensors store channels first.
    ax.imshow(tensor.permute(1, 2, 0).numpy())
    title = "Original photograph" if name == "original" else labels[name]
    ax.set_title(f"{title}\n{tensor.shape[2]} x {tensor.shape[1]} pixels", fontsize=11)
    ax.axis("off")
    if name != "original":
        transforms.ToPILImage()(tensor).save(OUTPUT_DIR / f"{name}.png")
    manifest["results"][name] = {"tensor_shape": list(tensor.shape), "width": tensor.shape[2], "height": tensor.shape[1]}
fig.suptitle("Schwartzberg, Paul | CSCI E-89 | HW5 Problem 1", fontsize=14)
fig.savefig(OUTPUT_DIR / "tiger_transformations.png", dpi=120)
plt.show()
(OUTPUT_DIR / "transformations.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Saved the five transformed images, comparison plot and transformation record in", OUTPUT_DIR)
