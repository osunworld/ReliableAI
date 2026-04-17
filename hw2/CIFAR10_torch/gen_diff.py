import argparse
import os

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from configs import bcolors
from models import DEFAULT_MODEL_IDS, load_models
from utils import (
    apply_transformation_constraint,
    averaged_coverage,
    clip_processed_image,
    init_coverage_tables,
    normalize,
    neuron_covered,
    neuron_to_cover,
    prediction_details,
    preprocess_raw_image,
    save_case,
    set_random_seed,
    update_coverage,
)

HW2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(HW2_DIR, "results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch DeepXplore-style differential testing on CIFAR-10 ResNet50 models."
    )
    # transformation / weight_* / step / threshold는 원본 DeepXplore의 핵심 탐색 파라미터에 해당한다.
    parser.add_argument("transformation", choices=["light", "occl", "blackout"])
    parser.add_argument("weight_diff", type=float)
    parser.add_argument("weight_nc", type=float)
    parser.add_argument("step", type=float)
    parser.add_argument("seeds", type=int)
    parser.add_argument("grad_iterations", type=int)
    parser.add_argument("threshold", type=float)
    parser.add_argument("--model1", default=DEFAULT_MODEL_IDS[0])
    parser.add_argument("--model2", default=DEFAULT_MODEL_IDS[1])
    parser.add_argument("--target_model", choices=[0, 1], default=0, type=int)
    parser.add_argument("--start_point", nargs=2, default=[0, 0], type=int)
    parser.add_argument("--occlusion_size", nargs=2, default=[8, 8], type=int)
    parser.add_argument("--output_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # 가능하면 GPU/MPS를 사용하고, 없으면 CPU에서 동일 코드가 동작하도록 한다.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 두 개의 오픈소스 ResNet50과 CIFAR-10 테스트셋을 준비한다.
    processor, models, class_names = load_models((args.model1, args.model2), device=device)
    dataset = CIFAR10(root="./data", train=False, download=True)

    # coverage table은 모델별 hidden state 채널 단위로 관리한다.
    sample_input = preprocess_raw_image(dataset[0][0], processor, device)
    coverage_tables = init_coverage_tables(models, sample_input)

    rng = np.random.default_rng(args.seed)
    successful_cases = 0

    for case_index in range(args.seeds):
        # seed 입력 하나를 고르고, 원본과 수정본 이미지를 각각 관리한다.
        sample_idx = int(rng.integers(0, len(dataset)))
        raw_image, true_label = dataset[sample_idx]

        gen_img = preprocess_raw_image(raw_image, processor, device).clone().detach()
        orig_img = gen_img.clone().detach()

        # 먼저 자연 입력만으로도 두 모델이 이미 불일치하는지 확인한다.
        with torch.no_grad():
            initial_outputs = [model(pixel_values=gen_img) for model in models]
        initial_predictions = [prediction_details(output.logits, class_names) for output in initial_outputs]
        predicted_labels = [prediction["label_id"] for prediction in initial_predictions]

        if predicted_labels[0] != predicted_labels[1]:
            for model, coverage_table in zip(models, coverage_tables):
                update_coverage(gen_img, model, coverage_table, args.threshold)

            print(
                bcolors.OKGREEN
                + "input already causes different outputs: {}, {}".format(
                    initial_predictions[0]["label_name"],
                    initial_predictions[1]["label_name"],
                )
                + bcolors.ENDC
            )

            save_case(
                args.output_dir,
                "already_differ_{:03d}".format(case_index),
                processor,
                orig_img,
                gen_img,
                true_label,
                {"items": initial_predictions, "class_names": class_names},
                coverage_tables,
                (args.model1, args.model2),
            )
            successful_cases += 1
            continue

        original_label = predicted_labels[0]

        # 자연 입력에서 불일치가 없으면 gradient ascent로 새로운 테스트 입력을 생성한다.
        for iteration in range(args.grad_iterations):
            gen_img.requires_grad_(True)
            outputs = [model(pixel_values=gen_img, output_hidden_states=True) for model in models]
            logits = [output.logits for output in outputs]
            target_neurons = [neuron_to_cover(coverage_table) for coverage_table in coverage_tables]

            # differential loss:
            # 원래 합의하던 클래스를 한 모델에서는 약하게, 다른 모델에서는 유지/강화하도록 유도한다.
            if args.target_model == 0:
                diff_loss = -args.weight_diff * logits[0][0, original_label] + logits[1][0, original_label]
            else:
                diff_loss = logits[0][0, original_label] - args.weight_diff * logits[1][0, original_label]

            # coverage loss:
            # 아직 덜 본 뉴런을 활성화하도록 hidden state 평균값을 손실에 더한다.
            coverage_loss = torch.tensor(0.0, device=device)
            for output, (layer_idx, channel_idx) in zip(outputs, target_neurons):
                coverage_loss = coverage_loss + output.hidden_states[layer_idx][0, channel_idx].mean()

            total_loss = diff_loss + args.weight_nc * coverage_loss

            for model in models:
                model.zero_grad(set_to_none=True)
            if gen_img.grad is not None:
                gen_img.grad.zero_()

            total_loss.backward()
            grads = normalize(gen_img.grad.detach())
            grads = apply_transformation_constraint(
                grads,
                args.transformation,
                tuple(args.start_point),
                tuple(args.occlusion_size),
            )
            gen_img = clip_processed_image((gen_img + grads * args.step).detach(), processor)

            # 새 입력으로 두 모델을 다시 평가해 disagreement가 생겼는지 확인한다.
            with torch.no_grad():
                updated_outputs = [model(pixel_values=gen_img) for model in models]
            updated_predictions = [prediction_details(output.logits, class_names) for output in updated_outputs]
            updated_labels = [prediction["label_id"] for prediction in updated_predictions]

            if updated_labels[0] != updated_labels[1]:
                for model, coverage_table in zip(models, coverage_tables):
                    update_coverage(gen_img, model, coverage_table, args.threshold)

                coverage_values = [neuron_covered(coverage_table)[2] for coverage_table in coverage_tables]
                print(
                    bcolors.OKGREEN
                    + "case {} diverged at iter {}: {} vs {} | coverage {:.3f}, {:.3f} | avg {:.3f}".format(
                        case_index,
                        iteration,
                        updated_predictions[0]["label_name"],
                        updated_predictions[1]["label_name"],
                        coverage_values[0],
                        coverage_values[1],
                        averaged_coverage(coverage_tables),
                    )
                    + bcolors.ENDC
                )

                save_case(
                    args.output_dir,
                    "generated_{:03d}".format(successful_cases),
                    processor,
                    orig_img,
                    gen_img,
                    true_label,
                    {"items": updated_predictions, "class_names": class_names},
                    coverage_tables,
                    (args.model1, args.model2),
                )
                successful_cases += 1
                break

    print(
        bcolors.OKBLUE
        + "Finished. Saved {} disagreement-inducing inputs to {}".format(successful_cases, args.output_dir)
        + bcolors.ENDC
    )


if __name__ == "__main__":
    main()
