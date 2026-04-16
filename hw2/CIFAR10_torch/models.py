import re

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL_IDS = (
    "jialicheng/cifar10_resnet-50",
    "jialicheng/unlearn-so_cifar10_resnet-50_salun_10_100",
)

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def safe_model_name(model_id):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", model_id)


def image_size_from_processor(processor):
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 224, 224


def load_models(model_ids=DEFAULT_MODEL_IDS, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_ids[0], use_fast=False)

    models = []
    for model_id in model_ids:
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.name = safe_model_name(model_id)
        model.to(device)
        model.eval()
        models.append(model)

    return processor, models, CIFAR10_LABELS
