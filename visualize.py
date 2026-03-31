import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsm import fgsm_targeted, fgsm_untargeted
from pgd import pgd_targeted, pgd_untargeted


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NUM_CLASSES = 10


def _tensor_to_numpy_image(tensor, dataset_name):
    image = tensor.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if dataset_name == "MNIST":
        image = np.squeeze(image)

    return np.clip(image, 0.0, 1.0)


def _perturbation_to_numpy_image(perturbation, dataset_name, scale_factor):
    noise = perturbation.detach().cpu().numpy()

    if noise.ndim == 3 and noise.shape[0] in (1, 3):
        noise = np.transpose(noise, (1, 2, 0))

    if dataset_name == "MNIST":
        noise = np.squeeze(noise)

    noise = noise * scale_factor
    noise = np.clip((noise + 1.0) / 2.0, 0.0, 1.0)
    return noise


def _generate_adversarial_images(model, images, labels, attack_method, targeted, eps, k, eps_step):
    targets = (labels + 1) % NUM_CLASSES if targeted else None

    if attack_method == "FGSM":
        if targeted:
            adv_images = fgsm_targeted(model, images, targets, eps)
        else:
            adv_images = fgsm_untargeted(model, images, labels, eps)
    elif attack_method == "PGD":
        if k is None or eps_step is None:
            raise ValueError("PGD visualization requires both k and eps_step.")
        if targeted:
            adv_images = pgd_targeted(model, images, targets, k, eps, eps_step)
        else:
            adv_images = pgd_untargeted(model, images, labels, k, eps, eps_step)
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")

    return adv_images, targets


def save_visualization(
    model,
    test_loader,
    device,
    dataset_name,
    attack_method,
    targeted,
    eps,
    k=None,
    eps_step=None,
    num_samples=5,
    perturbation_scale=10.0,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model.eval()

    successful_samples = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images, targets = _generate_adversarial_images(
            model=model,
            images=images,
            labels=labels,
            attack_method=attack_method,
            targeted=targeted,
            eps=eps,
            k=k,
            eps_step=eps_step,
        )

        with torch.no_grad():
            orig_outputs = model(images)
            adv_outputs = model(adv_images)

        orig_preds = orig_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)

        for idx in range(labels.size(0)):
            label = labels[idx].item()
            orig_pred = orig_preds[idx].item()
            adv_pred = adv_preds[idx].item()
            target = targets[idx].item() if targeted else None

            orig_correct = orig_pred == label
            adv_success = adv_pred == target if targeted else adv_pred != label

            if not (orig_correct and adv_success):
                continue

            successful_samples.append(
                {
                    "label": label,
                    "target": target,
                    "orig_pred": orig_pred,
                    "adv_pred": adv_pred,
                    "orig_image": images[idx].detach().cpu(),
                    "adv_image": adv_images[idx].detach().cpu(),
                    "perturbation": (adv_images[idx] - images[idx]).detach().cpu(),
                }
            )

            if len(successful_samples) >= num_samples:
                break

        if len(successful_samples) >= num_samples:
            break

    if len(successful_samples) < num_samples:
        raise RuntimeError(
            f"Only found {len(successful_samples)} successful samples for "
            f"{dataset_name} {attack_method} ({'targeted' if targeted else 'untargeted'})."
        )

    mode_name = "Targeted" if targeted else "Untargeted"
    file_stub = f"{dataset_name.lower().replace('-', '').replace(' ', '_')}_{mode_name.lower()}_{attack_method.lower()}"

    fig, axes = plt.subplots(num_samples, 3, figsize=(11, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    title = f"{dataset_name} - {mode_name} {attack_method} (eps={eps}"
    if attack_method == "PGD":
        title += f", k={k}, step={eps_step}"
    title += ")"
    fig.suptitle(title, fontsize=15)

    cmap = "gray" if dataset_name == "MNIST" else None

    for row, sample in enumerate(successful_samples):
        orig_image = _tensor_to_numpy_image(sample["orig_image"], dataset_name)
        adv_image = _tensor_to_numpy_image(sample["adv_image"], dataset_name)
        perturbation = _perturbation_to_numpy_image(
            sample["perturbation"], dataset_name, perturbation_scale
        )

        axes[row, 0].imshow(orig_image, cmap=cmap)
        axes[row, 0].set_title(f"Original\ntrue={sample['label']}, pred={sample['orig_pred']}")
        axes[row, 0].axis("off")

        adv_title = f"Adversarial\npred={sample['adv_pred']}"
        if targeted:
            adv_title += f", target={sample['target']}"
        else:
            adv_title += f", true={sample['label']}"
        axes[row, 1].imshow(adv_image, cmap=cmap)
        axes[row, 1].set_title(adv_title)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(perturbation, cmap=cmap)
        axes[row, 2].set_title(f"Perturbation x{perturbation_scale:g}")
        axes[row, 2].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_path = os.path.join(RESULTS_DIR, f"{file_stub}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

    return output_path
