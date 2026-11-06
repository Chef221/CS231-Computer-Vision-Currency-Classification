"""Small convolutional networks with explicit spatial stages and initialization."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.10),
        )


class CurrencyCNN(nn.Module):
    """RGB/gray -> convolution blocks -> GAP -> dense128 -> six logits."""

    def __init__(self, in_channels=3, channels=(32, 64, 128), n_classes=6):
        super().__init__()
        stages = []
        previous = in_channels
        for width in channels:
            stages.append(ConvBlock(previous, width))
            previous = width
        self.blocks = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 128), nn.ReLU(), nn.Dropout(0.50),
            nn.Linear(128, n_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def feature_maps(self, images):
        for block in self.blocks:
            images = block(images)
        return images

    def forward(self, images):
        maps = self.feature_maps(images)
        return self.head(self.pool(maps).flatten(1))

    def embed(self, images):
        return self.pool(self.feature_maps(images)).flatten(1)


def build_model(config):
    return CurrencyCNN(1 if config["color"] == "gray" else 3, config["channels"])


def architecture_table(config):
    """Shapes/parameters are derived from the actual modules, without training."""
    model = build_model(config).eval()
    rows = []
    x = torch.zeros(1, 1 if config["color"] == "gray" else 3, 224, 224)
    with torch.no_grad():
        for index, block in enumerate(model.blocks, 1):
            x = block(x)
            rows.append({"stage": f"block_{index}", "output": str(tuple(x.shape[1:])),
                         "parameters": sum(p.numel() for p in block.parameters())})
        x = model.pool(x).flatten(1)
        rows.append({"stage": "global_average_pool", "output": str(tuple(x.shape[1:])),
                     "parameters": 0})
        rows.append({"stage": "dense128_dropout_dense6", "output": "(6,)",
                     "parameters": sum(p.numel() for p in model.head.parameters())})
    return rows
