"""The VGG16 flower architecture and checkpoint loader for HW5.

Author: Schwartzberg, Paul.
The original ImageNet classifier is replaced with a five-class flower head.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models


class VGGFlowerCNN(nn.Module):
    """VGG16 features followed by a small flower classifier."""

    def __init__(self, num_classes=5, image_size=150, pretrained=False):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        original = models.vgg16(weights=weights)
        self.features = original.features
        del original  # Release the large ImageNet classifier.
        self.image_size = image_size
        # Five 2-by-2 pools: 150 -> 75 -> 37 -> 18 -> 9 -> 4.
        self.flattened_features = 512 * (image_size // 32) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def load_vgg_checkpoint(path, device="cpu"):
    """Rebuild the architecture without downloading ImageNet weights."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = VGGFlowerCNN(
        num_classes=len(checkpoint["class_names"]),
        image_size=checkpoint["image_size"], pretrained=False,
    )
    if model.flattened_features != checkpoint["flattened_features"]:
        raise ValueError("The saved feature size does not match the architecture.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
