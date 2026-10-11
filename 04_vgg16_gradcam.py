"""Schwartzberg, Paul — Problem 4: Grad-CAM for the stage 2 flower model.

Run after Problem 3 has saved outputs/problem3/stage2_vgg16.pth.
This inspects one fixed validation photograph without changing model weights.
"""

# %% 1. Load the validation-selected stage 2 model.
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from hw5_common import prepare_flower_data, seed_everything, write_json
from hw5_vgg import load_vgg_checkpoint

seed_everything(42)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")
OUTPUT_DIR = Path("outputs/problem4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path("outputs/problem3/stage2_vgg16.pth")
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError("Run Problem 3 first to save its stage 2 checkpoint.")

model, checkpoint = load_vgg_checkpoint(CHECKPOINT_PATH, device=DEVICE)
model.eval()
model.requires_grad_(False)
CLASS_NAMES = checkpoint["class_names"]
IMAGE_SIZE = int(checkpoint["image_size"])
MEAN = checkpoint["mean"]
STD = checkpoint["std"]

# Keep every trained parameter unchanged. Gradients will track the input.
original_state = {
    name: value.detach().cpu().clone()
    for name, value in model.state_dict().items()
}

# VGG uses in-place ReLU by default. Disable that memory optimisation so that
# the last convolution's output remains available to differentiate exactly.
# Evaluation predictions are unchanged: both ReLU forms compute max(0, x).
for layer in model.modules():
    if isinstance(layer, nn.ReLU):
        layer.inplace = False
TARGET_LAYER_INDEX = 28
assert isinstance(model.features[TARGET_LAYER_INDEX], nn.Conv2d)
print("Python:", sys.executable)
print("Loaded:", CHECKPOINT_PATH)
print("Stage:", checkpoint["stage"], "| best epoch:", checkpoint["best_epoch"])
print("Target convolution: features[28] (VGG16 conv5_3)")
print("Classes:", CLASS_NAMES)

# %% 2. Choose the photograph before examining the prediction.
flower_data = prepare_flower_data()
assert flower_data["class_names"] == CLASS_NAMES
DAISY_LABEL = CLASS_NAMES.index("daisy")
daisy_records = sorted(
    (path, label)
    for path, label in flower_data["records"]["validation"]
    if int(label) == DAISY_LABEL
)
assert daisy_records, "No daisy photographs in the saved validation split."
IMAGE_PATH, TRUE_LABEL = daisy_records[0]

# Match the resize and ImageNet normalisation used for the stage 2 model.
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
with Image.open(IMAGE_PATH) as photograph:
    original_image = photograph.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    image_tensor = image_transform(photograph.convert("RGB"))
display_image = np.asarray(original_image).astype(np.float32) / 255.0
with torch.no_grad():
    expected_logits = model(image_tensor.unsqueeze(0).to(DEVICE))
    probabilities = expected_logits.softmax(dim=1)[0].cpu()
PREDICTED_LABEL = int(probabilities.argmax())
PREDICTED_SCORE = float(probabilities[PREDICTED_LABEL])
print("Fixed validation photograph:", IMAGE_PATH)
print("True flower:", CLASS_NAMES[int(TRUE_LABEL)])
print("Predicted flower:", CLASS_NAMES[PREDICTED_LABEL],
      f"({PREDICTED_SCORE:.1%})")

# %% 3. Calculate Grad-CAM at one convolution, following Lab 02's method.
def generate_gradcam(input_tensor, target_class):
    """Weight the target convolution's maps by class-logit gradients."""
    # A fresh input enables gradients without enabling weight updates.
    tracked_input = input_tensor.detach().unsqueeze(0).to(DEVICE)
    tracked_input.requires_grad_(True)

    # Stop at the final convolution, before its ReLU and the last max pool.
    activation_maps = model.features[:TARGET_LAYER_INDEX + 1](tracked_input)
    later_features = model.features[TARGET_LAYER_INDEX + 1:](activation_maps)
    logits = model.classifier(later_features)
    torch.testing.assert_close(logits.detach(), expected_logits,
                               rtol=1e-5, atol=1e-6)
    assert tuple(activation_maps.shape) == (1, 512, 9, 9)

    # Compute gradients of the chosen raw class score, not of cross-entropy.
    gradients = torch.autograd.grad(
        outputs=logits[0, int(target_class)],
        inputs=activation_maps,
    )[0]
    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    coarse_cam = torch.relu(
        (channel_weights * activation_maps).sum(dim=1, keepdim=True)
    ).detach()
    assert tuple(coarse_cam.shape) == (1, 1, 9, 9)
    positive_evidence = bool(coarse_cam.max().item() > 0)

    # Enlargement makes the coarse 9 x 9 map align with the displayed image.
    enlarged_cam = torch.nn.functional.interpolate(
        coarse_cam, size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False,
    )[0, 0].cpu()
    # Match Lab 02's min-max scaling; an all-zero map stays all zero.
    heatmap = ((enlarged_cam - enlarged_cam.min()) /
               (enlarged_cam.max() - enlarged_cam.min() + 1e-8))
    assert torch.isfinite(heatmap).all()
    return heatmap.numpy(), coarse_cam[0, 0].cpu().numpy(), positive_evidence


# The primary map explains the predicted class, as in Lab 02. If incorrect,
# also inspect the true class without replacing the original photograph.
target_classes = [PREDICTED_LABEL]
if int(TRUE_LABEL) != PREDICTED_LABEL:
    target_classes.append(int(TRUE_LABEL))
cam_results = {}
for target_class in target_classes:
    heatmap, coarse_map, has_positive_evidence = generate_gradcam(
        image_tensor, target_class
    )
    class_name = CLASS_NAMES[target_class]
    cam_results[class_name] = {
        "heatmap": heatmap,
        "coarse_map": coarse_map,
        "positive_evidence": has_positive_evidence,
    }
    np.save(OUTPUT_DIR / f"gradcam_{class_name}.npy", heatmap)
    np.save(OUTPUT_DIR / f"gradcam_{class_name}_coarse.npy", coarse_map)
    print(f"Target {class_name}: coarse map (9, 9); "
          f"positive evidence: {has_positive_evidence}")

# %% 4. Display and save the evidence, then verify the weights.
fig, axes = plt.subplots(
    len(cam_results), 3, figsize=(13, 4.2 * len(cam_results)), squeeze=False
)
for row, (class_name, result) in enumerate(cam_results.items()):
    heatmap = result["heatmap"]
    axes[row, 0].imshow(display_image)
    axes[row, 0].set_title(
        f"True: {CLASS_NAMES[int(TRUE_LABEL)]}\n"
        f"Predicted: {CLASS_NAMES[PREDICTED_LABEL]} ({PREDICTED_SCORE:.1%})"
    )
    plotted_map = axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[row, 1].set_title(f"Grad-CAM target: {class_name}\nRelative intensity")
    axes[row, 2].imshow(display_image)
    axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[row, 2].set_title(f"Overlay for {class_name}\nfeatures[28], coarse 9 x 9")
    for axis in axes[row]:
        axis.axis("off")
    fig.colorbar(plotted_map, ax=axes[row, 1], fraction=0.046, pad=0.04)
fig.suptitle("Stage 2 VGG16: Grad-CAM — Schwartzberg, Paul", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUTPUT_DIR / "vgg16_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

# Exact equality detects accidental changes to any stored parameter or buffer.
for name, value in model.state_dict().items():
    torch.testing.assert_close(value.detach().cpu(), original_state[name],
                               rtol=0, atol=0)
assert all(parameter.grad is None for parameter in model.parameters())
with torch.no_grad():
    torch.testing.assert_close(
        model(image_tensor.unsqueeze(0).to(DEVICE)), expected_logits,
        rtol=0, atol=0,
    )
record = {
    "author": "Schwartzberg, Paul",
    "checkpoint": CHECKPOINT_PATH.as_posix(),
    "stage": checkpoint["stage"],
    "best_epoch": int(checkpoint["best_epoch"]),
    "selection": "First sorted daisy path from the saved validation split",
    "image": Path(IMAGE_PATH).relative_to(flower_data["root"]).as_posix(),
    "dataset": "Flower Photos",
    "true_class": CLASS_NAMES[int(TRUE_LABEL)],
    "predicted_class": CLASS_NAMES[PREDICTED_LABEL],
    "predicted_score": PREDICTED_SCORE,
    "probabilities": {
        name: float(value) for name, value in zip(CLASS_NAMES, probabilities)
    },
    "target_layer": "features[28] (VGG16 conv5_3, a single Conv2d)",
    "activation_shape": [1, 512, 9, 9],
    "coarse_heatmap_shape": [9, 9],
    "display_shape": [IMAGE_SIZE, IMAGE_SIZE],
    "cam_targets": {
        name: {"positive_evidence": result["positive_evidence"]}
        for name, result in cam_results.items()
    },
    "weights_unchanged": True,
    "method": "Lab 02 Grad-CAM; direct class-logit gradients via autograd.grad",
}
write_json(OUTPUT_DIR / "gradcam_results.json", record)
print("All model weights and buffers are unchanged; no training took place.")
print("Saved the plot, numerical maps and result record to", OUTPUT_DIR)
