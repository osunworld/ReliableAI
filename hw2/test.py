import os
import sys

import torch
from torchvision.datasets import CIFAR10

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_DEEPXPLORE_DIR = os.path.join(CURRENT_DIR, "CIFAR10_torch")
if TORCH_DEEPXPLORE_DIR not in sys.path:
    sys.path.insert(0, TORCH_DEEPXPLORE_DIR)

from models import DEFAULT_MODEL_IDS, load_models
from utils import prediction_details, preprocess_raw_image


def main():
    # smoke test:
    # 모델 2개가 정상 로드되고 CIFAR-10 입력에 대해 예측을 내는지만 빠르게 확인한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Loading CIFAR-10 sample and PyTorch models ===")
    print("Using device:", device)
    processor, models, class_names = load_models(DEFAULT_MODEL_IDS, device=device)

    dataset = CIFAR10(root="./data", train=False, download=True)
    sample_image, true_label = dataset[0]
    sample_tensor = preprocess_raw_image(sample_image, processor, device)

    predictions = []
    for model in models:
        with torch.no_grad():
            outputs = model(pixel_values=sample_tensor)
        predictions.append(prediction_details(outputs.logits, class_names))

    print("True label:", class_names[true_label])
    for model_id, prediction in zip(DEFAULT_MODEL_IDS, predictions):
        print(
            "{} -> {} ({:.4f})".format(
                model_id,
                prediction["label_name"],
                prediction["confidence"],
            )
        )


if __name__ == "__main__":
    main()
