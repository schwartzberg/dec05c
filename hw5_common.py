"""Schwartzberg, Paul. Shared data and evaluation tools for CSCI E-89 HW5.

All flower problems reuse the same split. This module never trains a model on
validation or test images. Training is requested explicitly through run_epoch.
"""
from pathlib import Path
import json
import random
import shutil
import hashlib

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_score

ROOT = Path(__file__).resolve().parent
SEED = 42
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def seed_everything(seed=SEED):
    """Reset Python, NumPy and PyTorch random choices for a repeatable run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def choose_device():
    """Prefer CUDA if present; this computer currently uses the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_json(path, data):
    """Write a readable UTF-8 record of settings or measured results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def cpu_state(model):
    """Copy weights and buffers: later updates cannot alter this checkpoint."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def prepare_flower_data(data_dir=ROOT / "data"):
    """Find/download Flower Photos and save a fixed stratified 70/15/15 split.

    A local copy from HW4 can supply the original downloaded files. Fresh clones
    download the archive if neither copy is present. Only the paths and labels,
    not resized images or augmentation, determine the split.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    flower_root = data_dir / "flower_photos"
    if not flower_root.exists():
        prior = ROOT.parents[2] / "Week4" / "Assignment" / "dec04c" / "data" / "flower_photos"
        if prior.exists():
            print("Copying the original Flower Photos files already downloaded for HW4.", flush=True)
            shutil.copytree(prior, flower_root)
        else:
            download_and_extract_archive(FLOWER_URL, download_root=str(data_dir))
    catalog = datasets.ImageFolder(flower_root)
    records = [(Path(path), int(label)) for path, label in catalog.samples]
    class_names = list(catalog.classes)
    assert class_names == ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    train, remaining = train_test_split(records, test_size=0.30, random_state=SEED,
                                        stratify=[label for _, label in records])
    validation, test = train_test_split(remaining, test_size=0.50, random_state=SEED,
                                        stratify=[label for _, label in remaining])
    splits = {"train": train, "validation": validation, "test": test}
    path_sets = [set(path for path, _ in rows) for rows in splits.values()]
    assert all(path_sets[i].isdisjoint(path_sets[j]) for i in range(3) for j in range(i+1,3))
    assert set.union(*path_sets) == {path for path, _ in records}
    manifest = {
        "author": "Schwartzberg, Paul", "seed": SEED, "dataset_url": FLOWER_URL,
        "class_names": class_names, "split_proportions": [0.70, 0.15, 0.15],
        "counts": {name: len(rows) for name, rows in splits.items()},
        "class_counts": {name: np.bincount([label for _, label in rows], minlength=5).tolist()
                         for name, rows in splits.items()},
        "splits": {name: [path.relative_to(flower_root).as_posix() for path, _ in rows]
                   for name, rows in splits.items()},
    }
    # A digest identifies this exact ordering and split in every saved experiment.
    manifest["split_sha256"] = hashlib.sha256(
        json.dumps(manifest["splits"], sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = ROOT / "outputs/shared/split_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("The saved flower split differs from the current dataset. Preserve and review it before training.")
    write_json(manifest_path, manifest)
    return {"root": flower_root, "class_names": class_names, "records": splits, "manifest": manifest}

class PhotoDataset(Dataset):
    """Open each RGB photo and apply only this split's transform.

    Optional image caching stores original RGB pixels, before random training
    transforms. Augmentation is still sampled again for each visit.
    """
    def __init__(self, records, transform, cache_images=False):
        self.records = list(records)
        self.transform = transform
        self.cached = None
        if cache_images:
            self.cached = []
            for path, _ in self.records:
                with Image.open(path) as picture:
                    self.cached.append(picture.convert("RGB"))
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        path, label = self.records[index]
        if self.cached is None:
            with Image.open(path) as picture:
                image = self.transform(picture.convert("RGB"))
        else:
            image = self.transform(self.cached[index].copy())
        return image, label

def run_epoch(model, loader, device, optimizer=None):
    """Train when an optimizer is supplied; otherwise evaluate without gradients."""
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite cross-entropy loss.")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(labels)
    assert count == len(loader.dataset), "The loader omitted or repeated dataset items."
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}

def evaluate_classifier(model, loader, device, class_names):
    """Evaluate every supplied image; report accuracy and the literal precision metric."""
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    actual, predicted, probabilities = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            loss_sum += float(loss_function(logits, labels.to(device))) * len(labels)
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1).cpu().tolist())
    count = len(actual)
    assert count == len(loader.dataset), "Final evaluation must use the entire dataset."
    labels = list(range(len(class_names)))
    report = classification_report(actual, predicted, labels=labels, target_names=class_names,
                                   zero_division=0, output_dict=True)
    return {
        "count": count, "loss": loss_sum / count,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "report": report, "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "true_labels": actual, "predicted_labels": predicted, "probabilities": probabilities,
    }
