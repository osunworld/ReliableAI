import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import image_size_from_processor


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _channel_stats(values, device):
    if not isinstance(values, (list, tuple)):
        values = [values] * 3
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return tensor.view(1, 3, 1, 1)


def preprocess_raw_image(raw_image, processor, device):
    if isinstance(raw_image, Image.Image):
        image = np.array(raw_image)
    else:
        image = np.array(raw_image)

    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    height, width = image_size_from_processor(processor)
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)

    mean = _channel_stats(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), device)
    std = _channel_stats(getattr(processor, "image_std", [0.229, 0.224, 0.225]), device)
    return (tensor - mean) / std


def deprocess_image(processed_image, processor):
    image = processed_image.detach().cpu()
    if image.dim() == 4:
        image = image[0]

    mean = _channel_stats(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), image.device)
    std = _channel_stats(getattr(processor, "image_std", [0.229, 0.224, 0.225]), image.device)
    image = image.unsqueeze(0) * std + mean
    image = image.clamp(0.0, 1.0)
    image = (image[0].permute(1, 2, 0) * 255.0).round().byte().numpy()
    return image


def save_image(array, path):
    Image.fromarray(array).save(path)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def clip_processed_image(processed_image, processor):
    image = processed_image
    mean = _channel_stats(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), image.device)
    std = _channel_stats(getattr(processor, "image_std", [0.229, 0.224, 0.225]), image.device)
    raw = (image * std + mean).clamp(0.0, 1.0)
    return (raw - mean) / std


def normalize(x):
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = torch.zeros_like(gradients)
    row, col = start_point
    height, width = rect_shape
    new_grads[:, :, row:row + height, col:col + width] = gradients[:, :, row:row + height, col:col + width]
    return new_grads


def constraint_light(gradients):
    grad_mean = torch.mean(gradients)
    return torch.ones_like(gradients) * grad_mean


def constraint_black(gradients, rect_shape=(10, 10)):
    start_row = random.randint(0, gradients.shape[2] - rect_shape[0])
    start_col = random.randint(0, gradients.shape[3] - rect_shape[1])
    new_grads = torch.zeros_like(gradients)
    patch = gradients[:, :, start_row:start_row + rect_shape[0], start_col:start_col + rect_shape[1]]
    if torch.mean(patch) < 0:
        new_grads[:, :, start_row:start_row + rect_shape[0], start_col:start_col + rect_shape[1]] = -torch.ones_like(
            patch
        )
    return new_grads


def apply_transformation_constraint(gradients, transformation, start_point, occlusion_size):
    if transformation == "light":
        return constraint_light(gradients)
    if transformation == "occl":
        return constraint_occl(gradients, start_point, occlusion_size)
    if transformation == "blackout":
        return constraint_black(gradients, rect_shape=occlusion_size)
    raise ValueError("Unsupported transformation: {}".format(transformation))


def scale(intermediate_layer_output, rmax=1.0, rmin=0.0):
    max_val = intermediate_layer_output.max()
    min_val = intermediate_layer_output.min()
    if max_val == min_val:
        return np.zeros_like(intermediate_layer_output)
    standardized = (intermediate_layer_output - min_val) / (max_val - min_val)
    return standardized * (rmax - rmin) + rmin


def _hidden_states(model, input_tensor):
    outputs = model(pixel_values=input_tensor, output_hidden_states=True)
    return outputs.hidden_states


def init_coverage_tables(models, sample_input):
    coverage_tables = []
    with torch.no_grad():
        for model in models:
            hidden_states = _hidden_states(model, sample_input)
            model_layer_dict = defaultdict(bool)
            for layer_idx, hidden_state in enumerate(hidden_states):
                channels = int(hidden_state.shape[1])
                for channel_idx in range(channels):
                    model_layer_dict[(layer_idx, channel_idx)] = False
            coverage_tables.append(model_layer_dict)
    return coverage_tables


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), covered in model_layer_dict.items() if not covered]
    if not_covered:
        return random.choice(not_covered)
    return random.choice(list(model_layer_dict.keys()))


def neuron_covered(model_layer_dict):
    covered_neurons = len([value for value in model_layer_dict.values() if value])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0.0):
    with torch.no_grad():
        hidden_states = _hidden_states(model, input_data)
        for layer_idx, hidden_state in enumerate(hidden_states):
            scaled = scale(hidden_state[0].detach().cpu().numpy())
            for channel_idx in range(scaled.shape[0]):
                if np.mean(scaled[channel_idx]) > threshold and not model_layer_dict[(layer_idx, channel_idx)]:
                    model_layer_dict[(layer_idx, channel_idx)] = True


def prediction_details(logits, class_names):
    probabilities = torch.softmax(logits, dim=-1)
    label_id = int(torch.argmax(probabilities[0]).item())
    confidence = float(probabilities[0, label_id].item())
    return {
        "label_id": label_id,
        "label_name": class_names[label_id],
        "confidence": confidence,
    }


def averaged_coverage(coverage_tables):
    covered = sum(neuron_covered(table)[0] for table in coverage_tables)
    total = sum(neuron_covered(table)[1] for table in coverage_tables)
    return covered / float(total)


def save_case(output_dir, case_prefix, processor, orig_img, gen_img, true_label, predictions, coverage_tables, model_ids):
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, case_prefix + "_orig.png")
    generated_path = os.path.join(output_dir, case_prefix + "_gen.png")
    meta_path = os.path.join(output_dir, case_prefix + ".json")

    save_image(deprocess_image(orig_img, processor), original_path)
    save_image(deprocess_image(gen_img, processor), generated_path)

    metadata = {
        "model_ids": model_ids,
        "true_label": predictions["class_names"][true_label],
        "predictions": predictions["items"],
        "coverage": [neuron_covered(table)[2] for table in coverage_tables],
        "average_coverage": averaged_coverage(coverage_tables),
        "original_image": os.path.basename(original_path),
        "generated_image": os.path.basename(generated_path),
    }
    save_json(metadata, meta_path)
