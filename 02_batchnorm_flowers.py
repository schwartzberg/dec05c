"""Schwartzberg, Paul. Run all four HW5 Problem 2 flower experiments."""
from hw5_common import *
from hw5_basic_cnn import *

# Show the environment and use four CPU threads per training process.
import time
import sys
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.metrics import ConfusionMatrixDisplay

torch.set_num_threads(4)
seed_everything()
device = choose_device()
IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE = 64, 64, 30, 0.001
OUTPUT_DIR = ROOT / "outputs/problem2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__, "| device:", device)
print("Settings:", IMAGE_SIZE, "pixels; batch", BATCH_SIZE, "; epochs", EPOCHS, "; Adam learning rate", LEARNING_RATE)


# Prepare one split for every model, and show its class counts.
flower_data = prepare_flower_data()
flower_classes = flower_data["class_names"]
flower_records = flower_data["records"]
for name in ("train", "validation", "test"):
    counts = flower_data["manifest"]["class_counts"][name]
    print(name, len(flower_records[name]), "images:", dict(zip(flower_classes, counts)))

# Validation and test receive only the fixed resize and pixel scaling.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
])
augmented_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])
val_dataset = PhotoDataset(flower_records["validation"], eval_transform, cache_images=True)
test_dataset = PhotoDataset(flower_records["test"], eval_transform, cache_images=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                         generator=torch.Generator().manual_seed(SEED))


# Show four random training views of one image, with the same label.
preview_path, preview_label = flower_records["train"][0]
with Image.open(preview_path) as picture:
    picture = picture.convert("RGB")
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), layout="constrained")
    for index, ax in enumerate(axes):
        shown = (eval_transform if index == 0 else augmented_transform)(picture)
        ax.imshow(shown.permute(1, 2, 0))
        ax.set_title("Resize only" if index == 0 else f"Training view {index}")
        ax.axis("off")
fig.suptitle(f"Schwartzberg, Paul | Training augmentation: {flower_classes[preview_label]}")
fig.savefig(OUTPUT_DIR / "augmentation_examples.png", dpi=120)
plt.show()
# Experiments below reset all seeds after this preview.


# The only two experimental changes are BatchNorm and training augmentation.
EXPERIMENTS = [
    {"name": "original_no_augmentation", "batch_norm": False, "augmentation": False},
    {"name": "batchnorm_no_augmentation", "batch_norm": True, "augmentation": False},
    {"name": "original_with_augmentation", "batch_norm": False, "augmentation": True},
    {"name": "batchnorm_with_augmentation", "batch_norm": True, "augmentation": True},
]
seed_everything()
reference_model = BasicCNN(len(flower_classes))
INITIAL_DIGEST = shared_weight_digest(reference_model)
example_model = BasicCNN(len(flower_classes), batch_norm=True)
copy_shared_start(example_model, reference_model)
assert shared_weight_digest(example_model) == INITIAL_DIGEST
assert sum(isinstance(layer, nn.BatchNorm2d) for layer in example_model.modules()) == 3
for index, layer in enumerate(example_model.features):
    if isinstance(layer, nn.Conv2d):
        assert isinstance(example_model.features[index + 1], nn.BatchNorm2d)
print(example_model)
print("Shared initial convolution/linear digest:", INITIAL_DIGEST)


def train_basic_experiment(config):
    """Train one condition, using validation loss to select its checkpoint."""
    seed_everything()
    model = BasicCNN(len(flower_classes), batch_norm=config["batch_norm"]).to(device)
    copy_shared_start(model, reference_model)
    assert shared_weight_digest(model) == INITIAL_DIGEST
    transform = augmented_transform if config["augmentation"] else eval_transform
    dataset = PhotoDataset(flower_records["train"], transform, cache_images=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Reset stochastic transforms after construction; each run has the same batch order.
    seed_everything()
    history = {key: [] for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_loss, best_epoch, best_state = float("inf"), None, None
    start = time.perf_counter()
    print("Training:", config["name"], flush=True)
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, loader, device, optimizer)
        validation = run_epoch(model, val_loader, device)
        history["train_loss"].append(train["loss"])
        history["train_accuracy"].append(train["accuracy"])
        history["val_loss"].append(validation["loss"])
        history["val_accuracy"].append(validation["accuracy"])
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = cpu_state(model)
        # All epochs go to JSON; a short output keeps the submitted notebook readable.
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:02d}/{EPOCHS}: train loss {train['loss']:.4f}, acc {train['accuracy']:.2%}; "
                  f"validation loss {validation['loss']:.4f}, acc {validation['accuracy']:.2%}", flush=True)
    model.load_state_dict(best_state, strict=True)
    selected = run_epoch(model, val_loader, device)
    checkpoint = {
        "author": "Schwartzberg, Paul", "state_dict": cpu_state(model), "config": config,
        "class_names": flower_classes, "image_size": IMAGE_SIZE, "best_epoch": best_epoch,
        "seed": SEED, "split_sha256": flower_data["manifest"]["split_sha256"],
        "initial_shared_sha256": INITIAL_DIGEST, "selection_rule": "minimum validation cross-entropy loss",
    }
    checkpoint_path = OUTPUT_DIR / (config["name"] + ".pth")
    torch.save(checkpoint, checkpoint_path)
    # Confirm the saved file restores the selected model's scores, including BN buffers.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded = BasicCNN(len(flower_classes), batch_norm=config["batch_norm"]).to(device)
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    model.eval(); reloaded.eval()
    sample = val_dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        torch.testing.assert_close(model(sample), reloaded(sample), rtol=0, atol=0)
    result = {
        "author": "Schwartzberg, Paul", "config": config, "seed": SEED,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "initial_shared_sha256": INITIAL_DIGEST, "split_sha256": saved["split_sha256"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "selected_validation": selected, "history": history,
        "training_seconds": time.perf_counter() - start, "reload_verified": True,
    }
    write_json(OUTPUT_DIR / (config["name"] + "_metrics.json"), result)
    print(f"  Restored epoch {best_epoch}; validation loss {selected['loss']:.4f}; reload verified.", flush=True)
    return result

# Train and select all four checkpoints before examining any test results.
p2_results = [train_basic_experiment(config) for config in EXPERIMENTS]


# Evaluate every selected checkpoint on the same complete test set.
for result in p2_results:
    config = result["config"]
    checkpoint_path = OUTPUT_DIR / (config["name"] + ".pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BasicCNN(len(flower_classes), batch_norm=config["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test = evaluate_classifier(model, test_loader, device, flower_classes)
    assert test["count"] == 551 == flower_data["manifest"]["counts"]["test"]
    result["test"] = test
    write_json(OUTPUT_DIR / (config["name"] + "_metrics.json"), result)
    print(config["name"], f"test accuracy {test['accuracy']:.2%}; macro precision {test['macro_precision']:.2%}; "
          f"loss {test['loss']:.4f}; images {test['count']}")
write_json(OUTPUT_DIR / "comparison.json", {"author": "Schwartzberg, Paul", "experiments": p2_results})

# Accuracy is the fraction correct; macro precision gives each class equal weight.
rows = ["| Condition | Selected epoch | Validation loss | Test accuracy | Test macro precision |",
        "|---|---:|---:|---:|---:|"]
for result in p2_results:
    rows.append(f"| {result['config']['name']} | {result['best_epoch']} | {result['selected_validation']['loss']:.4f} "
                f"| {result['test']['accuracy']:.2%} | {result['test']['macro_precision']:.2%} |")
display(Markdown("\n".join(rows)))

# Show every run's learning history and selected validation checkpoint.
fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
for row, result in enumerate(p2_results):
    epochs = np.arange(1, len(result["history"]["train_loss"]) + 1)
    for col, metric in enumerate(("loss", "accuracy")):
        ax = axes[row, col]
        ax.plot(epochs, result["history"]["train_" + metric], label="Training")
        ax.plot(epochs, result["history"]["val_" + metric], label="Validation")
        ax.axvline(result["best_epoch"], color="grey", linestyle="--", label="Selected epoch")
        ax.set(title=result["config"]["name"], xlabel="Epoch", ylabel=metric.capitalize())
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        if metric == "accuracy": ax.set_ylim(0, 1)
fig.suptitle("Schwartzberg, Paul | Problem 2 learning histories")
fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=120)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
for ax, result in zip(axes.flat, p2_results):
    ConfusionMatrixDisplay(np.array(result["test"]["confusion_matrix"]), display_labels=flower_classes).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(result["config"]["name"], fontsize=10)
fig.suptitle("Schwartzberg, Paul | All 551 test images in each condition")
fig.savefig(OUTPUT_DIR / "test_confusion_matrices.png", dpi=120)
plt.show()
