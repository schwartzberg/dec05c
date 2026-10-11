# %% [markdown]
# # Problem 3: transfer learning with VGG16
# **Author: Schwartzberg, Paul.**
#
# I adapt the lecture notebook `3_FineTune_VGG16_PyTorch_Cat_Dog.ipynb`
# to the five flower classes. Stage 1 trains a new classifier while all
# ImageNet convolutional weights are frozen. Stage 2 starts from the selected
# stage-1 weights and fine-tunes only VGG16's final convolutional block and
# the classifier. Both selected stages are evaluated on the complete test set.
#
# Run all cells from top to bottom in the `cscie89` kernel. A fresh run downloads
# the official VGG16 weights (about 528 MB) once and then trains both stages.
# The CPU run uses four threads. Outputs and checkpoints are saved beneath
# `outputs/problem3`; these checkpoints are also used by Problem 4.

# %% [markdown]
# ## 1. Settings and the shared flower split
# The dataset contains 3,670 photographs, divided into 2,569 training, 550
# validation and 551 test images. The split is stratified by flower type and
# fixed by seed 42. Problem 2 uses this same split. The common module opens
# photographs, records the exact split, sets random seeds, and measures a
# complete pass through a loader. Its `run_epoch` uses integer class labels
# and `CrossEntropyLoss`, enabling updates only when an optimizer is supplied.

# %%
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from hw5_common import (
    ROOT, SEED, PhotoDataset, prepare_flower_data, seed_everything,
    choose_device, run_epoch, evaluate_classifier, write_json, cpu_state,
)
from hw5_vgg import VGGFlowerCNN, load_vgg_checkpoint

IMAGE_SIZE, BATCH_SIZE = 150, 20
INITIAL_EPOCHS, FINETUNE_EPOCHS = 5, 3
INITIAL_LR, FINETUNE_LR = 2e-5, 1e-5
PATIENCE = 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
seed_everything(SEED)
torch.set_num_threads(4)
device = choose_device()
torch.hub.set_dir(str(ROOT / "data" / "torch_hub"))
output_dir = ROOT / "outputs" / "problem3"
output_dir.mkdir(parents=True, exist_ok=True)
flower_data = prepare_flower_data()
class_names = flower_data["class_names"]
split_digest = flower_data["manifest"]["split_sha256"]
print("Device:", device)
print("Class names:", class_names)
print("Split counts:", flower_data["manifest"]["counts"])
print("Split SHA-256:", split_digest)

# %% [markdown]
# ## 2. Prepare RGB images for ImageNet weights
# Training images are resized to 150 by 150 pixels, randomly flipped left to
# right and rotated by up to 10 degrees. The random choices are sampled anew
# on each visit to an image. Validation and test images have no random changes.
# `ToTensor` scales pixel values into [0, 1]; ImageNet normalization subtracts
# each channel's mean and divides by its standard deviation. Labels remain
# integers from 0 to 4. Only the training loader shuffles images.

# %%
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# Cache original RGB pixels only; augmentation still changes on each visit.
train_data = PhotoDataset(flower_data["records"]["train"], train_transform, cache_images=True)
validation_data = PhotoDataset(flower_data["records"]["validation"], eval_transform, cache_images=True)
test_data = PhotoDataset(flower_data["records"]["test"], eval_transform, cache_images=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)
assert len(train_data) + len(validation_data) + len(test_data) == 3670
print("Image tensor shape:", tuple(train_data[0][0].shape), "| Label:", train_data[0][1])

# %% [markdown]
# ## 3. Keep VGG16 features and replace its classifier
# The architecture is defined in `hw5_vgg.py` so Problem 4 can reload it.
# `VGGFlowerCNN` retains torchvision's VGG16 `.features`, removes its large
# 1,000-class ImageNet classifier, and defines the new head shown below.
#
# ```python
# self.features = original_vgg16.features
# self.classifier = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(8192, 256),
#     nn.ReLU(inplace=False),
#     nn.Dropout(0.5),
#     nn.Linear(256, 5),
# )
# ```
# Five pooling operations reduce 150 -> 75 -> 37 -> 18 -> 9 -> 4 pixels.
# The final 512 channels each contain 4 by 4 values: 8,192 values per image.
# The new head produces five raw class scores. `CrossEntropyLoss` accepts
# these scores directly; softmax is used later to show probability estimates.
# Dropout is active during training and disabled during evaluation.

# %%
model = VGGFlowerCNN(num_classes=len(class_names), image_size=IMAGE_SIZE,
                     pretrained=True).to(device)
for parameter in model.features.parameters():
    parameter.requires_grad = False
with torch.no_grad():
    feature_shape = model.features(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)).shape
assert tuple(feature_shape) == (1, 512, 4, 4)
assert model.flattened_features == 8192
print("Convolution output:", tuple(feature_shape))
print("New classifier:", model.classifier)
print("Total parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("Stage-1 trainable parameters:", sum(parameter.numel() for parameter in model.parameters()
                                             if parameter.requires_grad))
original_feature_weights = cpu_state(model.features)

# %% [markdown]
# ## 4. Train with validation loss and preserve the best checkpoint
# This follows the lecture's settings: RMSprop with alpha 0.9 and epsilon
# 1e-7, at most five initial epochs and three fine-tuning epochs, patience
# three. Each stage creates a new optimizer containing only trainable weights.
# The incoming model is evaluated before the first update and is an epoch-0
# candidate. This lets stage 2 retain its stage-1 input if fine-tuning fails
# to improve validation loss. A saved copy prevents subsequent updates from
# changing the best weights. Test images play no part in this selection.

# %%
def fit_vgg(model, epochs, learning_rate, stage_name):
    """Train one stage and restore its lowest-validation-loss weights."""
    started = time.perf_counter()
    optimizer = torch.optim.RMSprop(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, alpha=0.9, eps=1e-7,
    )
    incoming = run_epoch(model, validation_loader, device)
    best_loss, best_epoch, waiting = incoming["loss"], 0, 0
    best_weights = cpu_state(model)
    history = {"train_loss": [], "validation_loss": [],
               "train_accuracy": [], "validation_accuracy": [], "epoch_seconds": []}
    print(f"{stage_name} incoming | Validation: {incoming['accuracy']:.2%}"
          f" | Loss: {incoming['loss']:.4f}", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_result = run_epoch(model, train_loader, device, optimizer)
        validation_result = run_epoch(model, validation_loader, device)
        history["train_loss"].append(train_result["loss"])
        history["validation_loss"].append(validation_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["validation_accuracy"].append(validation_result["accuracy"])
        history["epoch_seconds"].append(time.perf_counter() - epoch_started)
        if validation_result["loss"] < best_loss:
            best_loss = validation_result["loss"]
            best_weights = cpu_state(model)
            best_epoch, waiting = epoch, 0
        else:
            waiting += 1
        print(f"{stage_name} {epoch}/{epochs} | Train: {train_result['accuracy']:.2%}"
              f" | Validation: {validation_result['accuracy']:.2%}"
              f" | Val loss: {validation_result['loss']:.4f}"
              f" | Seconds: {history['epoch_seconds'][-1]:.1f}", flush=True)
        # Persist progress as well as the final result; this does not select on test data.
        write_json(output_dir / (stage_name + "_progress.json"), {
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history,
        })
        if waiting >= PATIENCE:
            print("Early stopping after three epochs without validation-loss improvement.", flush=True)
            break
    model.load_state_dict(best_weights)
    selected = run_epoch(model, validation_loader, device)
    print(f"{stage_name}: restored epoch {best_epoch}"
          f" | Validation: {selected['accuracy']:.2%} | Loss: {selected['loss']:.4f}", flush=True)
    return {"stage": stage_name, "learning_rate": learning_rate, "maximum_epochs": epochs,
            "best_epoch": best_epoch, "incoming_validation": incoming,
            "selected_validation": selected, "history": history,
            "training_seconds": time.perf_counter() - started}


def save_vgg_stage(model, result, filename):
    """Save learned state together with architecture and preprocessing details."""
    checkpoint = {
        "author": "Schwartzberg, Paul", "state_dict": cpu_state(model),
        "class_names": class_names, "image_size": IMAGE_SIZE, "mean": MEAN, "std": STD,
        "flattened_features": model.flattened_features, "stage": result["stage"],
        "best_epoch": result["best_epoch"], "split_sha256": split_digest,
        "pretrained_source": "torchvision VGG16_Weights.IMAGENET1K_V1",
        "selection_rule": "Lowest validation loss; incoming weights included as epoch 0",
        "settings": {"batch_size": BATCH_SIZE, "learning_rate": result["learning_rate"],
                     "optimizer": "RMSprop", "alpha": 0.9, "eps": 1e-7, "patience": PATIENCE},
    }
    torch.save(checkpoint, output_dir / filename)
    write_json(output_dir / (result["stage"] + "_training.json"), result)
    print("Saved:", filename, flush=True)

# %% [markdown]
# ## 5. Stage 1: train the new classifier
# All 13 VGG16 convolutions are frozen. Only the two new linear layers learn.
# The check compares every convolutional weight and bias against the original
# ImageNet values after training. The saved checkpoint contains the restored
# best epoch, which can differ from the last epoch printed.

# %%
stage1_result = fit_vgg(model, INITIAL_EPOCHS, INITIAL_LR, "stage1")
for name, value in model.features.state_dict().items():
    torch.testing.assert_close(value.detach().cpu(), original_feature_weights[name], rtol=0, atol=0)
save_vgg_stage(model, stage1_result, "stage1_vgg16.pth")
print("Stage 1 check: all pretrained convolutional weights stayed unchanged.")
del original_feature_weights

# %% [markdown]
# ## 6. Stage 2: fine-tune block 5 and the classifier
# `features[24:]` begins at VGG16 block 5, containing convolutions at indices
# 24, 26 and 28. Blocks 1-4 remain frozen. We continue with the selected
# stage-1 model and lower the learning rate from 2e-5 to 1e-5. A fresh optimizer
# includes the newly trainable parameters. The check below verifies that the
# earlier blocks remain exactly unchanged.

# %%
frozen_prefix_weights = cpu_state(model.features[:24])
for parameter in model.features[24:].parameters():
    parameter.requires_grad = True
print("Trainable convolution indices:", [index for index, layer in enumerate(model.features)
                                          if isinstance(layer, nn.Conv2d) and layer.weight.requires_grad])
print("Stage-2 trainable parameters:", sum(parameter.numel() for parameter in model.parameters()
                                             if parameter.requires_grad))
stage2_result = fit_vgg(model, FINETUNE_EPOCHS, FINETUNE_LR, "stage2")
for name, value in model.features[:24].state_dict().items():
    torch.testing.assert_close(value.detach().cpu(), frozen_prefix_weights[name], rtol=0, atol=0)
save_vgg_stage(model, stage2_result, "stage2_vgg16.pth")
print("Stage 2 check: VGG16 blocks 1-4 stayed unchanged.")
del frozen_prefix_weights

# %% [markdown]
# ## 7. Plot the training and validation histories
# Each stage has its own epoch numbers. Dashed vertical lines identify the
# selected epoch when it is greater than zero. Training metrics are measured
# during updates, with augmentation and dropout; validation uses fixed
# preprocessing and evaluation mode. Their difference is therefore not a
# pure measure of overfitting. The test set is still unused at this point.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, result in enumerate([stage1_result, stage2_result]):
    history = result["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    for column, metric in enumerate(["loss", "accuracy"]):
        ax = axes[row, column]
        ax.plot(epochs, history["train_" + metric], marker="o", label="Training")
        ax.plot(epochs, history["validation_" + metric], marker="o", label="Validation")
        if result["best_epoch"] > 0:
            ax.axvline(result["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
        ax.set(xlabel="Epoch within stage", ylabel=metric.capitalize(),
               title=result["stage"] + ": " + metric)
        ax.legend()
        if metric == "accuracy":
            ax.set_ylim(0, 1)
fig.suptitle("Schwartzberg, Paul: VGG16 flower learning curves")
fig.tight_layout()
fig.savefig(output_dir / "learning_curves.png", dpi=140)
plt.show()

# %% [markdown]
# ## 8. Evaluate both selected stages on all 551 test images
# The assignment asks for precision. We report the literal macro precision
# (the equal-weight average of the five class precisions), accuracy and each
# class's precision, recall and F1. Precision asks how many predictions of a
# class are correct; recall asks how many actual members of that class were
# found. The confusion matrices place actual labels in rows and predictions
# in columns. Both models have already been selected using validation loss.
#
# Reloading each checkpoint verifies that the saved state reproduces its
# predictions. The full test set is evaluated once for each selected stage.
# We do not change settings after seeing these results.

# %%
test_results = {}
stage_models = {}
sample_for_reload_check = test_data[0][0].unsqueeze(0).to(device)
for stage, filename in [("stage1", "stage1_vgg16.pth"), ("stage2", "stage2_vgg16.pth")]:
    restored, checkpoint = load_vgg_checkpoint(output_dir / filename, device)
    assert checkpoint["split_sha256"] == split_digest
    # Check the reconstruction against the exact saved tensors and a second reload.
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), checkpoint["state_dict"][name], rtol=0, atol=0)
    restored.eval()
    with torch.no_grad():
        expected_scores = restored(sample_for_reload_check)
        verified, _ = load_vgg_checkpoint(output_dir / filename, device)
        torch.testing.assert_close(verified(sample_for_reload_check), expected_scores, rtol=0, atol=0)
    del verified
    result = evaluate_classifier(restored, test_loader, device, class_names)
    assert result["count"] == 551
    test_results[stage] = result
    stage_models[stage] = restored
    write_json(output_dir / (stage + "_test_metrics.json"), result)
    print(f"{stage} | Entire test set: {result['count']} images"
          f" | Accuracy: {result['accuracy']:.2%}"
          f" | Macro precision: {result['macro_precision']:.2%}"
          f" | Loss: {result['loss']:.4f}")
    print(classification_report(result["true_labels"], result["predicted_labels"],
                                labels=list(range(5)), target_names=class_names,
                                digits=3, zero_division=0))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, stage in zip(axes, ["stage1", "stage2"]):
    display = ConfusionMatrixDisplay(test_results[stage]["confusion_matrix"], display_labels=class_names)
    # Convert the saved list to the array expected by matplotlib's matrix display.
    import numpy as np
    display.confusion_matrix = np.asarray(display.confusion_matrix)
    display.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(stage + ": all 551 test images")
fig.suptitle("Schwartzberg, Paul: VGG16 test confusion matrices")
fig.tight_layout()
fig.savefig(output_dir / "test_confusion_matrices.png", dpi=140)
plt.show()

# %% [markdown]
# ## 9. Predict one fixed test image from every class
# For each class we use its first test filename in sorted order, chosen without
# checking predictions. Both stages see the same five images. The displayed
# score is the softmax estimate for the predicted class, not a guarantee of
# correctness. This small gallery illustrates inference; the overall results
# above use all 551 test images.

# %%
gallery_records = [sorted((path, label) for path, label in flower_data["records"]["test"]
                           if label == class_index)[0] for class_index in range(5)]
gallery_results = []
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for column, (path, actual_label) in enumerate(gallery_records):
    with Image.open(path) as picture:
        original = picture.convert("RGB")
        tensor = eval_transform(original).unsqueeze(0).to(device)
        for row, stage in enumerate(["stage1", "stage2"]):
            with torch.no_grad():
                probabilities = stage_models[stage](tensor).softmax(1)[0]
            predicted = int(probabilities.argmax())
            score = float(probabilities[predicted])
            axes[row, column].imshow(original)
            axes[row, column].set_title(f"{stage}: true {class_names[actual_label]}\n"
                                         f"Predicted {class_names[predicted]} ({score:.1%})", fontsize=9)
            axes[row, column].axis("off")
            gallery_results.append({"stage": stage, "path": path.relative_to(flower_data["root"]).as_posix(),
                                    "true_class": class_names[actual_label],
                                    "predicted_class": class_names[predicted], "score": score})
fig.suptitle("Schwartzberg, Paul: one fixed test image per flower class")
fig.tight_layout()
fig.savefig(output_dir / "five_class_predictions.png", dpi=140)
plt.show()
write_json(output_dir / "five_class_predictions.json", gallery_results)
comparison = {
    "author": "Schwartzberg, Paul", "class_names": class_names, "split_sha256": split_digest,
    "selection_rule": "Lowest validation loss, including the incoming epoch-0 candidate",
    "stage1": {"training": stage1_result, "test": test_results["stage1"]},
    "stage2": {"training": stage2_result, "test": test_results["stage2"]},
}
write_json(output_dir / "comparison.json", comparison)
accuracy_change = test_results["stage2"]["accuracy"] - test_results["stage1"]["accuracy"]
precision_change = test_results["stage2"]["macro_precision"] - test_results["stage1"]["macro_precision"]
print(f"Stage 2 minus stage 1 test accuracy: {100 * accuracy_change:+.2f} percentage points")
print(f"Stage 2 minus stage 1 macro precision: {100 * precision_change:+.2f} percentage points")
print("Both checkpoints reloaded successfully. The reported results include every test image.")
print("Saved reports, plots and both stages to:", output_dir)

# %% [markdown]
# ## 10. Interpretation
#
# In this recorded run, stage 1 selected epoch 5 and stage 2 selected epoch 3 by their lowest validation loss. Fine-tuning reduced the selected validation loss from **0.5321 to 0.4404**, while validation accuracy rose from **81.27% to 84.00%**.
#
# | Selected stage | Test images | Correct predictions | Test accuracy | Macro precision | Test loss |
# |---|---:|---:|---:|---:|---:|
# | Stage 1: frozen ImageNet base | 551 | 480 | 87.11% | 87.49% | 0.4112 |
# | Stage 2: fine-tune block 5 and head | 551 | 492 | 89.29% | 89.87% | 0.2831 |
#
# Stage 2 made **12 more correct predictions**, increasing accuracy by **2.18 percentage points** and macro precision by **2.38 percentage points**. Fine-tuning helped overall in this split, though it did not improve every class. Rose recall fell from **80.2% to 77.1%**: 18 roses were predicted as tulips in stage 2, compared with 12 in stage 1. Tulip precision also fell from **82.3% to 80.6%**. Those errors remain in the full reports and confusion matrices.
#
# Both stages correctly identified the five fixed gallery images, one from each class. These examples were chosen by sorted filename before examining predictions. Their success is illustrative; the overall scores above use every test image. A softmax estimate rounded to 100.0% still does not guarantee correctness.
#
# Training accuracy rose faster than validation accuracy during fine-tuning, widening the gap. Validation loss nevertheless improved in all three fine-tuning epochs. I kept the lecture's small epoch budget and selected checkpoints by validation loss. Test results were not used to alter settings or select epochs. These are results from one fixed split and seed, not a claim that every run will reproduce the same scores.
#
# Both saved checkpoints passed exact state and prediction checks after reloading. Stage 1 preserved every pretrained convolutional parameter; stage 2 preserved blocks 1-4. Problem 4 uses the saved stage-2 checkpoint for Grad-CAM.
