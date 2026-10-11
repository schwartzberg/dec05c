"""Check the saved HW5 submission without retraining or running inference.

Author: Schwartzberg, Paul.
Run with ``python check_submission.py`` after executing all four notebooks.
The checks compare independent records: split paths, prediction vectors,
histories, checkpoint tensors and the executed notebook/HTML artifacts.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import re
import sys

import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AUTHOR = "Schwartzberg, Paul"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CHECKS = []


def require(condition, message):
    """Stop with an explicit explanation when an expected invariant fails."""
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-8):
    require(math.isfinite(float(actual)) and
            math.isclose(float(actual), float(expected), rel_tol=tolerance,
                         abs_tol=tolerance),
            f"{message}: saved {actual!r}; expected {expected!r}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_artifacts():
    """Check the combined submission, retaining support for the earlier layout."""
    main_path = ROOT / "E89_Schwartzberg_Paul_ProblemSet05_01.ipynb"
    main_notebook = nbformat.read(main_path, as_version=4) if main_path.is_file() else None
    combined = main_notebook is not None and main_notebook.metadata.get("hw5_layout") == "combined"
    if combined:
        paths = [(main_path, main_notebook, "Combined notebook")]
    else:
        # The earlier individual notebooks may be kept as working records.
        legacy_root = ROOT
        if not all((legacy_root / f"E89_Schwartzberg_Paul_ProblemSet05_{number:02d}.ipynb").is_file()
                   for number in range(1, 5)):
            legacy_root = ROOT / "individual_problems"
        paths = [(legacy_root / f"E89_Schwartzberg_Paul_ProblemSet05_{number:02d}.ipynb",
                  None, f"Problem {number}") for number in range(1, 5)]
    artifacts = []
    for index, (path, notebook, label) in enumerate(paths, 1):
        if notebook is None:
            notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        text = "\n".join(cell.source for cell in notebook.cells)
        require(AUTHOR in text, f"{label}: full author name is missing")
        code_cells = [cell for cell in notebook.cells
                      if cell.cell_type == "code" and cell.source.strip()]
        require(code_cells, f"{label}: no code cells")
        coverage = {}
        if combined:
            for number in range(1, 5):
                require(re.search(rf"^## Problem {number}:", text, flags=re.MULTILINE),
                        f"Combined notebook: Problem {number} heading is missing")
                coverage[number] = sum(cell.metadata.get("hw5_problem") == number
                                       for cell in code_cells)
            require(coverage == {1: 6, 2: 9, 3: 9, 4: 4} and len(code_cells) == 28,
                    "Combined notebook: expected all 28 code cells with 6/9/9/4 problem coverage")
            require(all(cell.metadata.get("hw5_problem") in (1, 2, 3, 4)
                        for cell in code_cells),
                    "Combined notebook: a code cell has no problem attribution")
        counts = []
        for cell_index, cell in enumerate(code_cells, 1):
            ast.parse(cell.source)
            require(cell.execution_count is not None,
                    f"{label}: code cell {cell_index} was not executed")
            require(not any(output.output_type == "error"
                            for output in cell.get("outputs", [])),
                    f"{label}: code cell {cell_index} has an error output")
            counts.append(int(cell.execution_count))
        require(counts == sorted(counts) and len(set(counts)) == len(counts),
                f"{label}: cells were not executed in increasing order")
        html_path = path.with_suffix(".html")
        html = html_path.read_text(encoding="utf-8")
        require(AUTHOR in html, f"{label}: HTML author name is missing")
        require("data:image/png;base64," in html,
                f"{label}: HTML has no embedded PNG output")
        artifact = {
            "layout": "combined" if combined else "individual",
            "problems": [1, 2, 3, 4] if combined else [index],
            "code_cells": len(code_cells),
            "notebook_sha256": file_digest(path),
            "html_sha256": file_digest(html_path),
        }
        if combined:
            require(all(f"Problem {number}:" in html for number in range(1, 5)),
                    "Combined HTML: not all four problem headings are present")
            artifact["code_cells_by_problem"] = coverage
        artifacts.append(artifact)
    if combined:
        CHECKS.append("The single combined notebook contains all four problems and all 28 code cells, executed in order without errors; its HTML contains the author, problem headings and embedded plots.")
    else:
        CHECKS.append("All four individual notebooks are valid, executed in order and free of error outputs; their HTML reports contain the author and embedded plots.")
    return artifacts


def check_split():
    """Verify coverage, non-overlap and the exact ordering recorded by every run."""
    manifest = read_json(ROOT / "outputs/shared/split_manifest.json")
    require(manifest["author"] == AUTHOR, "Split manifest author differs")
    require(manifest["class_names"] == CLASSES, "Split class order differs")
    require(manifest["seed"] == 42, "Split seed differs from 42")
    expected_counts = {"train": 2569, "validation": 550, "test": 551}
    require(manifest["counts"] == expected_counts, "Split counts differ")
    path_sets = []
    for name, expected in expected_counts.items():
        paths = manifest["splits"][name]
        require(len(paths) == expected, f"{name}: wrong number of split paths")
        require(len(set(paths)) == expected, f"{name}: repeated split paths")
        require(all(Path(path).parts[0] in CLASSES for path in paths),
                f"{name}: unknown class folder")
        labels = [CLASSES.index(Path(path).parts[0]) for path in paths]
        require(np.bincount(labels, minlength=5).tolist() ==
                manifest["class_counts"][name], f"{name}: class counts differ")
        path_sets.append(set(paths))
    require(all(path_sets[i].isdisjoint(path_sets[j])
                for i in range(3) for j in range(i + 1, 3)),
            "Training, validation and test paths overlap")
    require(len(set.union(*path_sets)) == 3670,
            "The three splits do not cover 3,670 distinct images")
    # If the dataset is present, coverage can also be checked against its files.
    flower_root = ROOT / "data/flower_photos"
    if flower_root.is_dir():
        dataset_paths = {path.relative_to(flower_root).as_posix()
                         for name in CLASSES
                         for path in (flower_root / name).glob("*")
                         if path.suffix.lower() in
                         {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}
        require(set.union(*path_sets) == dataset_paths,
                "Split paths do not exactly cover the local flower images")
    digest = hashlib.sha256(json.dumps(manifest["splits"], sort_keys=True)
                            .encode("utf-8")).hexdigest()
    require(digest == manifest["split_sha256"], "Split digest does not match its paths")
    CHECKS.append("The shared seed-42 split covers 3,670 images exactly once, with 2,569/550/551 non-overlapping train/validation/test records.")
    return manifest


def check_metrics(metrics, manifest, name):
    """Recalculate classification measures from every saved test prediction."""
    actual = np.asarray(metrics["true_labels"])
    predicted = np.asarray(metrics["predicted_labels"])
    expected = np.asarray([CLASSES.index(Path(path).parts[0])
                           for path in manifest["splits"]["test"]])
    require(metrics["count"] == 551 and actual.shape == (551,) and
            predicted.shape == (551,), f"{name}: test evaluation is incomplete")
    require(actual.dtype.kind in "iu" and predicted.dtype.kind in "iu",
            f"{name}: labels are not integers")
    require(np.array_equal(actual, expected),
            f"{name}: true labels do not match every ordered test split path")
    require(np.all((predicted >= 0) & (predicted < 5)),
            f"{name}: predicted class outside 0-4")
    counts = np.zeros((5, 5), dtype=np.int64)
    np.add.at(counts, (actual, predicted), 1)
    require(np.array_equal(counts, np.asarray(metrics["confusion_matrix"])),
            f"{name}: confusion matrix differs from its prediction vectors")
    accuracy = float(np.mean(actual == predicted))
    precision = np.divide(np.diag(counts), counts.sum(0),
                          out=np.zeros(5, dtype=float), where=counts.sum(0) != 0)
    recall = np.diag(counts) / counts.sum(1)
    close(metrics["accuracy"], accuracy, f"{name}: accuracy")
    close(metrics["macro_precision"], precision.mean(), f"{name}: macro precision")
    close(metrics["balanced_accuracy"], recall.mean(), f"{name}: balanced accuracy")
    report = metrics["report"]
    for index, class_name in enumerate(CLASSES):
        close(report[class_name]["precision"], precision[index],
              f"{name}: {class_name} precision")
        close(report[class_name]["recall"], recall[index],
              f"{name}: {class_name} recall")
        require(report[class_name]["support"] == int(counts[index].sum()),
                f"{name}: {class_name} support differs")
    close(report["accuracy"], accuracy, f"{name}: report accuracy")
    close(report["macro avg"]["precision"], precision.mean(),
          f"{name}: report macro precision")
    probabilities = np.asarray(metrics["probabilities"])
    require(probabilities.shape == (551, 5) and np.isfinite(probabilities).all(),
            f"{name}: probability records are incomplete or non-finite")
    require(np.all(probabilities >= 0) and np.all(probabilities <= 1) and
            np.allclose(probabilities.sum(1), 1, atol=1e-6),
            f"{name}: probability rows do not sum to one")
    require(np.array_equal(probabilities.argmax(1), predicted),
            f"{name}: saved probabilities disagree with predictions")
    require(math.isfinite(metrics["loss"]) and metrics["loss"] >= 0,
            f"{name}: invalid test loss")
    return {"count": 551, "accuracy": accuracy,
            "macro_precision": float(precision.mean())}


def check_problem2(manifest):
    comparison = read_json(ROOT / "outputs/problem2/comparison.json")
    require(comparison["author"] == AUTHOR, "Problem 2 author differs")
    results = comparison["experiments"]
    require(len(results) == 4, "Problem 2 requires four experiments")
    combinations = {(result["config"]["batch_norm"], result["config"]["augmentation"])
                    for result in results}
    require(combinations == {(False, False), (True, False),
                             (False, True), (True, True)},
            "Problem 2 does not contain each BatchNorm/augmentation combination")
    initial_digests = {result["initial_shared_sha256"] for result in results}
    require(len(initial_digests) == 1 and len(next(iter(initial_digests))) == 64,
            "Problem 2 shared starting-weight digests differ")
    summaries = {}
    for result in results:
        name = result["config"]["name"]
        require(result["split_sha256"] == manifest["split_sha256"],
                f"{name}: different data split")
        require(result["seed"] == 42 and result["epochs"] == 30 and
                result["batch_size"] == 64 and result["learning_rate"] == 0.001,
                f"{name}: experiment settings differ")
        history = result["history"]
        require(all(len(history[key]) == 30 for key in
                    ("train_loss", "val_loss", "train_accuracy", "val_accuracy")),
                f"{name}: incomplete epoch history")
        expected_epoch = int(np.argmin(history["val_loss"])) + 1
        require(result["best_epoch"] == expected_epoch,
                f"{name}: selected epoch does not minimise validation loss")
        close(result["selected_validation"]["loss"],
              history["val_loss"][expected_epoch - 1], f"{name}: selected validation loss")
        close(result["selected_validation"]["accuracy"],
              history["val_accuracy"][expected_epoch - 1], f"{name}: selected validation accuracy")
        require(result["selected_validation"]["count"] == 550 and result["reload_verified"],
                f"{name}: validation or reload record differs")
        checkpoint = torch.load(ROOT / f"outputs/problem2/{name}.pth",
                                map_location="cpu", weights_only=True)
        require(checkpoint["config"] == result["config"] and
                checkpoint["best_epoch"] == expected_epoch and
                checkpoint["split_sha256"] == manifest["split_sha256"] and
                checkpoint["initial_shared_sha256"] in initial_digests,
                f"{name}: checkpoint metadata differs from measured results")
        state = checkpoint["state_dict"]
        running_means = [key for key in state if key.endswith("running_mean")]
        require(len(running_means) == (3 if result["config"]["batch_norm"] else 0),
                f"{name}: wrong number of BatchNorm state buffers")
        if result["config"]["batch_norm"]:
            # Each selected epoch includes 41 batches, including the final nine images.
            for index, width in zip((1, 5, 9), (16, 32, 64)):
                require(tuple(state[f"features.{index}.running_mean"].shape) == (width,),
                        f"{name}: BatchNorm channel count differs")
                require(int(state[f"features.{index}.num_batches_tracked"]) ==
                        expected_epoch * math.ceil(2569 / 64),
                        f"{name}: BatchNorm statistics were not preserved at the selected epoch")
        summaries[name] = check_metrics(result["test"], manifest, name)
    CHECKS.append("Problem 2 contains all four controlled conditions, matching initial weights and split digests, complete 30-epoch histories, validation-selected checkpoints and recomputed full-test metrics.")
    return summaries


def check_vgg_history(training, name):
    history = training["history"]
    losses = [training["incoming_validation"]["loss"]] + history["validation_loss"]
    accuracies = [training["incoming_validation"]["accuracy"]] + history["validation_accuracy"]
    expected_epoch = int(np.argmin(losses))
    require(training["best_epoch"] == expected_epoch,
            f"{name}: selected epoch does not minimise validation loss including epoch 0")
    close(training["selected_validation"]["loss"], losses[expected_epoch],
          f"{name}: selected validation loss")
    close(training["selected_validation"]["accuracy"], accuracies[expected_epoch],
          f"{name}: selected validation accuracy")
    require(training["incoming_validation"]["count"] == 550 and
            training["selected_validation"]["count"] == 550,
            f"{name}: validation did not cover all 550 images")
    count = len(history["validation_loss"])
    require(1 <= count <= training["maximum_epochs"] and
            all(len(history[key]) == count for key in
                ("train_loss", "train_accuracy", "validation_accuracy", "epoch_seconds")),
            f"{name}: inconsistent epoch histories")
    return expected_epoch


def check_problem3(manifest):
    comparison = read_json(ROOT / "outputs/problem3/comparison.json")
    require(comparison["author"] == AUTHOR and comparison["class_names"] == CLASSES and
            comparison["split_sha256"] == manifest["split_sha256"],
            "Problem 3 author, classes or split differ")
    checkpoints, summaries = {}, {}
    for name in ("stage1", "stage2"):
        result = comparison[name]
        selected_epoch = check_vgg_history(result["training"], name)
        summaries[name] = check_metrics(result["test"], manifest, name)
        checkpoint = torch.load(ROOT / f"outputs/problem3/{name}_vgg16.pth",
                                map_location="cpu", weights_only=True)
        require(checkpoint["stage"] == name and checkpoint["best_epoch"] == selected_epoch and
                checkpoint["class_names"] == CLASSES and checkpoint["image_size"] == 150 and
                checkpoint["flattened_features"] == 8192 and
                checkpoint["split_sha256"] == manifest["split_sha256"],
                f"{name}: VGG checkpoint metadata differs")
        require(checkpoint["mean"] == [0.485, 0.456, 0.406] and
                checkpoint["std"] == [0.229, 0.224, 0.225],
                f"{name}: ImageNet preprocessing differs")
        state = checkpoint["state_dict"]
        require(tuple(state["classifier.1.weight"].shape) == (256, 8192) and
                tuple(state["classifier.4.weight"].shape) == (5, 256),
                f"{name}: classifier does not produce five flower scores")
        checkpoints[name] = checkpoint
    official_path = ROOT / "data/torch_hub/checkpoints/vgg16-397923af.pth"
    official = torch.load(official_path, map_location="cpu", weights_only=True)
    stage1, stage2 = (checkpoints[name]["state_dict"] for name in ("stage1", "stage2"))
    feature_keys = [name for name in stage1 if name.startswith("features.")]
    require(len(feature_keys) == 26, "VGG16 should have weights and biases for 13 convolutions")
    for name in feature_keys:
        require(torch.equal(stage1[name], official[name]),
                f"Stage 1 changed frozen ImageNet tensor {name}")
        if int(name.split(".")[1]) < 24:
            require(torch.equal(stage1[name], stage2[name]),
                    f"Stage 2 changed a frozen tensor in blocks 1-4: {name}")
    late_changed = any(not torch.equal(stage1[name], stage2[name])
                       for name in feature_keys if int(name.split(".")[1]) >= 24)
    if checkpoints["stage2"]["best_epoch"] > 0:
        require(late_changed, "A selected fine-tuning epoch did not change any block-5 tensor")
    else:
        require(all(torch.equal(value, stage2[name]) for name, value in stage1.items()),
                "Stage 2 selected epoch 0 but does not equal the incoming stage-1 model")
    gallery = read_json(ROOT / "outputs/problem3/five_class_predictions.json")
    for stage in ("stage1", "stage2"):
        rows = [row for row in gallery if row["stage"] == stage]
        require(len(rows) == 5 and {row["true_class"] for row in rows} == set(CLASSES),
                f"{stage}: prediction gallery is missing a flower class")
        for row in rows:
            candidates = sorted(path for path in manifest["splits"]["test"]
                                if Path(path).parts[0] == row["true_class"])
            require(row["path"] == candidates[0],
                    f"{stage}: gallery image was not the first sorted test filename")
            index = manifest["splits"]["test"].index(row["path"])
            predicted = comparison[stage]["test"]["predicted_labels"][index]
            require(row["predicted_class"] == CLASSES[predicted],
                    f"{stage}: gallery prediction differs from full test evaluation")
    CHECKS.append("Both VGG stages use the entire test split and validation-selected histories; stage 1 preserves official ImageNet features, stage 2 preserves blocks 1-4, and each gallery covers all five classes.")
    return summaries, checkpoints["stage2"]


def check_problem4(stage2, manifest):
    record = read_json(ROOT / "outputs/problem4/gradcam_results.json")
    require(record["author"] == AUTHOR and record["stage"] == "stage2" and
            record["best_epoch"] == stage2["best_epoch"],
            "Grad-CAM does not identify the selected stage-2 checkpoint")
    require(record["checkpoint"] == "outputs/problem3/stage2_vgg16.pth" and
            record["target_layer"] == "features[28] (VGG16 conv5_3, a single Conv2d)",
            "Grad-CAM must target one convolution of the stage-2 model")
    require(record["weights_unchanged"] is True,
            "Grad-CAM did not record unchanged weights")
    require("direct class-logit gradients via autograd.grad" in record["method"],
            "Grad-CAM gradient method is not recorded")
    require(record["activation_shape"] == [1, 512, 9, 9] and
            record["coarse_heatmap_shape"] == [9, 9] and record["display_shape"] == [150, 150],
            "Grad-CAM target/display dimensions differ")
    candidates = sorted(path for path in manifest["splits"]["validation"]
                        if Path(path).parts[0] == "daisy")
    require(record["image"] == candidates[0] and record["true_class"] == "daisy",
            "Grad-CAM image does not match the preselected validation daisy")
    require(record["predicted_class"] in record["cam_targets"],
            "Grad-CAM does not explain the predicted class")
    for name, detail in record["cam_targets"].items():
        require(name in CLASSES, "Grad-CAM target is not a flower class")
        heatmap = np.load(ROOT / f"outputs/problem4/gradcam_{name}.npy")
        coarse = np.load(ROOT / f"outputs/problem4/gradcam_{name}_coarse.npy")
        require(heatmap.shape == (150, 150) and coarse.shape == (9, 9) and
                np.isfinite(heatmap).all() and np.isfinite(coarse).all(),
                f"{name}: Grad-CAM arrays are incomplete or non-finite")
        require(np.min(heatmap) >= 0 and np.max(heatmap) <= 1 and np.min(coarse) >= 0,
                f"{name}: Grad-CAM contains values outside its declared range")
        require(detail["positive_evidence"] == bool(np.max(coarse) > 0),
                f"{name}: positive-evidence record differs from the saved map")
    require((ROOT / "outputs/problem4/vgg16_gradcam.png").is_file(),
            "The Grad-CAM figure is missing")
    CHECKS.append("Grad-CAM identifies the selected stage-2 model and single conv5_3 target, uses the fixed validation image and records finite numerical maps, class targets and unchanged weights.")
    return {"image": record["image"], "target_layer": record["target_layer"],
            "cam_targets": list(record["cam_targets"]), "weights_unchanged": True}


def main():
    artifacts = check_artifacts()
    manifest = check_split()
    problem2 = check_problem2(manifest)
    problem3, stage2 = check_problem3(manifest)
    problem4 = check_problem4(stage2, manifest)
    report = {
        "author": AUTHOR, "validated_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "checks": CHECKS, "artifacts": artifacts,
        "split_sha256": manifest["split_sha256"],
        "problem2": problem2, "problem3": problem3, "problem4": problem4,
    }
    destination = ROOT / "outputs/submission_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All saved submission checks passed.")
    print("Validation report:", destination)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, ValueError, SyntaxError) as error:
        print("Submission validation failed:", error, file=sys.stderr)
        sys.exit(1)
