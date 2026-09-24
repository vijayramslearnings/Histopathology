"""Deterministic and augmented image preprocessing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    train: bool = False
    scale_fractions: tuple[float, float, float] = (1.0, 0.75, 0.55)
    rotation_degrees: float = 15.0
    brightness_contrast: float = 0.1


class MultiScalePreprocessor:
    """Create 5x/10x/20x context approximations from one image.

    The fractions define how much of the shorter image dimension is retained.
    Larger fractions represent larger fields of view, not verified physical
    magnifications.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.config.train:
            return image
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.5:
            image = ImageOps.flip(image)
        if self.config.rotation_degrees:
            angle = random.uniform(
                -self.config.rotation_degrees, self.config.rotation_degrees
            )
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        amount = self.config.brightness_contrast
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(1.0 - amount, 1.0 + amount)
        )
        return ImageEnhance.Contrast(image).enhance(
            random.uniform(1.0 - amount, 1.0 + amount)
        )

    def _crop(self, image: Image.Image, fraction: float) -> Image.Image:
        width, height = image.size
        side = max(1, min(width, height, round(min(width, height) * fraction)))
        max_left = max(0, width - side)
        max_top = max(0, height - side)
        left = random.randint(0, max_left) if self.config.train else max_left // 2
        top = random.randint(0, max_top) if self.config.train else max_top // 2
        return image.crop((left, top, left + side, top + side))

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize(
            (self.config.image_size, self.config.image_size), Image.Resampling.BICUBIC
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = self._augment(image)
        tensors = [
            self._to_tensor(self._crop(image, fraction))
            for fraction in self.config.scale_fractions
        ]
        return torch.stack(tensors)


class BaselinePreprocessor(MultiScalePreprocessor):
    """Return one full-context tensor for the ResNet baseline."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        multi = super().__call__(image)
        return multi[-1]


def build_preprocessor(
    image_size: int,
    train: bool,
    scale_fractions: Sequence[float] = (1.0, 0.75, 0.55),
) -> MultiScalePreprocessor:
    return MultiScalePreprocessor(
        PreprocessConfig(
            image_size=image_size,
            train=train,
            scale_fractions=tuple(float(value) for value in scale_fractions),
        )
    )