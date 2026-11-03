"""Explicit preprocessing policies shared by datasets and pretrained adapters."""
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

PREPROCESSING = {"version": 1, "tile_size": 256, "model_size": 224, "resize": "bilinear_antialias_square",
                 "mean": [.485, .456, .406], "std": [.229, .224, .225],
                 "note": "Shared full-tile resize for comparison; no model-specific center crop or stain normalization"}


def validate_image_settings(*, task, batch_size, num_workers, max_pixels, max_patches_per_bag,
                            image_size, max_samples_per_split, mean, std, class_names):
    if task not in {"classification", "segmentation", "mil"}:
        raise ValueError("task must be classification, segmentation or mil")
    if batch_size < 1 or num_workers < 0 or max_pixels < 1 or max_patches_per_bag < 1:
        raise ValueError("Invalid data resource bounds")
    if image_size is not None and (type(image_size) is not int or image_size < 32):
        raise ValueError("image_size must be null or an integer >= 32")
    if image_size is not None and image_size ** 2 > max_pixels:
        raise ValueError("Resized model input exceeds max_pixels")
    if max_samples_per_split is not None and max_samples_per_split < 1:
        raise ValueError("max_samples_per_split must be positive or null")
    if len(mean) != 3 or len(std) != 3 or not np.isfinite([*mean, *std]).all() or min(std) <= 0:
        raise ValueError("Expected three finite normalization means and positive standard deviations")
    if task == "mil" and batch_size != 1:
        raise ValueError("Variable-size MIL bags require data.batch_size=1; use gradient accumulation")
    if task != "segmentation" and (not class_names or len(class_names) < 2 or len(set(class_names)) != len(class_names)):
        raise ValueError("Set ordered, unique class_names (at least two)")


def transform_patch(rgb, *, image_size, mean, std, augment, target=None, valid=None):
    if image_size is not None:
        size = [image_size, image_size]
        rgb = TF.resize(rgb, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if target is not None:
            target = TF.resize(target, size, interpolation=InterpolationMode.NEAREST)
            valid = TF.resize(valid, size, interpolation=InterpolationMode.NEAREST)
    if augment:
        for axis in (-1, -2):
            if torch.rand(()) < 0.5:
                rgb = rgb.flip(axis)
                if target is not None:
                    target, valid = target.flip(axis), valid.flip(axis)
    return TF.normalize(rgb, mean, std), target, valid


def matched_encoder_transform():
    return transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
                               transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])


def contrastive_transform():
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ColorJitter(.2, .2, .2, .02),
        transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])
