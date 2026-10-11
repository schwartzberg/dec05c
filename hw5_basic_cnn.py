"""Schwartzberg, Paul. BasicCNN variants for CSCI E-89 HW5 Problem 2."""
import hashlib
import torch
from torch import nn

class BasicCNN(nn.Module):
    """The three-block course model, with optional BatchNorm after each Conv2d."""
    def __init__(self, num_classes=5, batch_norm=False, image_size=64):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (16, 32, 64):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))  # Before ReLU, after every convolution.
            layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * (image_size // 8) ** 2, 64),
            nn.ReLU(), nn.Linear(64, num_classes),
        )
    def forward(self, images):
        return self.classifier(self.features(images))

def shared_layers(model):
    """Return the convolutions and linear layers, in the same order in both variants."""
    return [layer for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear))]

def copy_shared_start(model, reference):
    """Give every experiment exactly the same starting convolution and linear weights."""
    for target, source in zip(shared_layers(model), shared_layers(reference), strict=True):
        target.load_state_dict(source.state_dict(), strict=True)

def shared_weight_digest(model):
    """Identify the shared initial weights, excluding the added BatchNorm parameters."""
    digest = hashlib.sha256()
    for layer in shared_layers(model):
        for value in layer.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
