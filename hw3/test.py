#!/usr/bin/env python3
"""
작은 외부 CNN을 학습하고 ONNX로 내보낸 뒤,
Marabou로 local robustness를 검증하는 단일 실행 스크립트.

이 파일 안에서 과제의 핵심 절차 1~4를 모두 수행한다.

1. 기준 입력 x 선택
2. 입력 제약 ||x' - x||∞ <= epsilon 설정
3. 출력 제약(원래 클래스 유지) 설정
4. Marabou 실행 및 SAT/UNSAT 해석
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def _require_module(name: str, install_hint: str):
    # 실행에 꼭 필요한 패키지가 없으면 즉시 종료하고 설치 방법을 알려준다.
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency '{name}'. Install it with:\n{install_hint}"
        ) from exc


torch = _require_module(
    "torch",
    "pip install torch onnx onnxruntime scikit-learn numpy",
)
onnx = _require_module(
    "onnx",
    "pip install onnx onnxruntime",
)
onnxruntime = _require_module(
    "onnxruntime",
    "pip install onnx onnxruntime",
)

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
MARABOU_ROOT = ROOT / "Marabou"
if not MARABOU_ROOT.exists():
    raise SystemExit(f"Marabou repository not found at {MARABOU_ROOT}")

sys.path.insert(0, str(MARABOU_ROOT))
from maraboupy import Marabou


class SmallDigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Marabou가 다루기 쉬운 작은 모델
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2-layer CNN on sklearn digits and verify local robustness with Marabou."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=10, help="Seconds per Marabou query")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional index inside the held-out test split. Must be correctly classified.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # 실험 재현성을 위해 난수 시드를 고정한다.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def prepare_data(seed: int):
    # sklearn digits는 8x8 이미지라서 MNIST보다 훨씬 작고,
    # CNN 구조를 쓰면서도 Marabou 검증 시간을 줄이기 좋다.
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    x = np.expand_dims(x, axis=1)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def train_model(model: nn.Module, x_train, y_train, epochs: int, batch_size: int, learning_rate: float):
    # 학습 데이터는 TensorDataset으로 감싸고 minibatch 학습을 수행한다.
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_examples = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_examples += xb.size(0)

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            avg_loss = total_loss / total_examples
            print(f"[train] epoch={epoch:02d} loss={avg_loss:.4f}")


def evaluate_model(model: nn.Module, x_test, y_test):
    # test accuracy와 margin을 같이 구해두면
    # "모델이 자신 있게 맞춘 샘플"을 기준 입력으로 고르기 쉽다.
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test)).cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    accuracy = float((predictions == y_test).mean())
    margins = np.partition(logits, -2, axis=1)[:, -1] - np.partition(logits, -2, axis=1)[:, -2]
    print(f"[eval] test_accuracy={accuracy:.4f}")
    return logits, predictions, margins


def select_sample(x_test, y_test, logits, predictions, margins, sample_index: int | None):
    # Step 1. 검증의 기준이 될 입력 x를 선택한다.
    #인자 없으면 correctly classified sample 중 margin이 가장 큰 것을 선택
    if sample_index is not None:
        if not (0 <= sample_index < len(x_test)):
            raise SystemExit(f"--sample-index must be between 0 and {len(x_test) - 1}")
        if predictions[sample_index] != y_test[sample_index]:
            raise SystemExit(
                f"sample {sample_index} is misclassified "
                f"(pred={predictions[sample_index]}, true={y_test[sample_index]})."
            )
        return sample_index

    correct_indices = np.where(predictions == y_test)[0]
    if len(correct_indices) == 0:
        raise SystemExit("No correctly classified test sample was found.")
    best_local_index = int(np.argmax(margins[correct_indices]))
    return int(correct_indices[best_local_index])


def export_to_onnx(model: nn.Module, onnx_path: Path) -> None:
    # Marabou가 읽을 수 있도록 학습한 PyTorch 모델을 ONNX로 export한다.
    # 마지막 softmax는 넣지 않고 logits를 그대로 사용한다.
    dummy_input = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(onnx_path))


def apply_input_constraints(network, sample: np.ndarray, epsilon: float):
    """
    Step 2. 입력 제약을 설정
    각 픽셀 x_i에 대해 max(0, x_i - epsilon) <= x'_i <= min(1, x_i + epsilon)
    형태의 box constraint를 건다.
    이는 ||x' - x||∞ <= epsilon 을 각 좌표별 bound로 풀어쓴 것
    """
    input_vars = np.array(network.inputVars[0]).flatten()
    flat_sample = sample.flatten()

    for idx, var in enumerate(input_vars):
        lower = float(max(0.0, flat_sample[idx] - epsilon))
        upper = float(min(1.0, flat_sample[idx] + epsilon))
        network.setLowerBound(var, lower)
        network.setUpperBound(var, upper)

    return input_vars


def add_output_constraint(network, output_vars: np.ndarray, true_label: int, target_label: int):
    """
    Step 3. 출력 제약 설정
    output_true >= output_target 이 항상 유지되는지 여부를 확인하기 위해
    output_true - output_target <= 0 가능한지 검사
    가능하면 SAT: target_label 쪽으로 뒤집히는 반례가 존재
    불가능하면 UNSAT: 그 target에 대해서는 안전
    """
    network.addInequality(
        [output_vars[true_label], output_vars[target_label]],
        [1.0, -1.0],
        0.0,
        isProperty=True,
    )


def run_marabou_query(network, timeout: int):
    """
    Step 4. Marabou를 실행한다.
    timeout은 target class 하나를 검사할 때 허용하는 최대 시간이다.
    """
    options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=0)
    started = time.perf_counter()
    exit_code, vals, _ = network.solve(verbose=False, options=options)
    elapsed = time.perf_counter() - started
    return exit_code, vals, elapsed


def verify_target(
    onnx_path: Path,
    sample: np.ndarray,
    true_label: int,
    target_label: int,
    epsilon: float,
    timeout: int,
    ort_session,
):
    # 매 target class마다 새 network를 읽어오는 이유는
    # 속성 제약이 서로 섞이지 않게 하기 위해서이다.
    network = Marabou.read_onnx(str(onnx_path))
    output_vars = np.array(network.outputVars[0]).flatten()

    input_vars = apply_input_constraints(network, sample, epsilon)
    add_output_constraint(network, output_vars, true_label, target_label)
    exit_code, vals, elapsed = run_marabou_query(network, timeout)

    result = {
        "target_label": target_label,
        "exit_code": exit_code,
        "elapsed": elapsed,
    }

    if exit_code == "sat":
        # SAT이면 Marabou가 실제 반례 입력 x'를 돌려준다.
        # 이를 다시 ONNX runtime에 넣어 실제 예측이 바뀌는지 확인한다.
        adv_input = np.array([vals[var] for var in input_vars], dtype=np.float32).reshape(1, 1, 8, 8)
        adv_logits = ort_session.run(None, {"input": adv_input})[0][0]
        result["adv_input"] = adv_input[0, 0]
        result["adv_logits"] = adv_logits
        result["adv_pred"] = int(np.argmax(adv_logits))
        result["linf"] = float(np.max(np.abs(adv_input.reshape(-1) - sample.reshape(-1))))

    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("[setup] loading sklearn digits dataset")
    x_train, x_test, y_train, y_test = prepare_data(args.seed)

    print("[setup] training small 2-layer CNN")
    model = SmallDigitsCNN()
    train_model(model, x_train, y_train, args.epochs, args.batch_size, args.learning_rate)

    logits, predictions, margins = evaluate_model(model, x_test, y_test)

    # Step 1. 먼저 기준 입력 x를 선택한다.
    sample_index = select_sample(x_test, y_test, logits, predictions, margins, args.sample_index)
    sample = x_test[sample_index]
    true_label = int(y_test[sample_index])
    predicted_label = int(predictions[sample_index])
    sample_margin = float(margins[sample_index])

    print(
        "[sample] "
        f"index={sample_index} true_label={true_label} predicted_label={predicted_label} margin={sample_margin:.4f}"
    )
    print(
        "[property] "
        f"verify that all x' with ||x' - x||_inf <= {args.epsilon:.4f} keep label {true_label}"
    )

    with tempfile.TemporaryDirectory(prefix="marabou_hw3_") as tmp_dir:
        # ONNX 파일은 제출물이 아니라 중간 산출물이므로 임시 디렉터리에 둔다.
        onnx_path = Path(tmp_dir) / "external_digits_cnn.onnx"
        print(f"[export] writing external ONNX model to {onnx_path}")
        export_to_onnx(model, onnx_path)

        ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_logits = ort_session.run(None, {"input": sample[np.newaxis, ...].astype(np.float32)})[0][0]
        onnx_pred = int(np.argmax(onnx_logits))
        print(f"[export] onnx_predicted_label={onnx_pred}")

        print(
            "[verify] "
            f"checking local robustness with epsilon={args.epsilon:.4f} and timeout={args.timeout}s per target"
        )
        results = []
        for target_label in range(10):
            if target_label == true_label:
                continue

            # 각 target class마다 입력 제약, 출력 제약, Marabou 실행을 반복한다.
            result = verify_target(
                onnx_path=onnx_path,
                sample=sample,
                true_label=true_label,
                target_label=target_label,
                epsilon=args.epsilon,
                timeout=args.timeout,
                ort_session=ort_session,
            )
            results.append(result)
            print(
                "[verify] "
                f"target={target_label} result={result['exit_code']} runtime={result['elapsed']:.2f}s"
            )
            if result["exit_code"] == "sat":
                # SAT이 나오면 이미 반례를 찾은 것
                print(
                    "[counterexample] "
                    f"target={target_label} adv_pred={result['adv_pred']} linf={result['linf']:.6f}"
                )
                print("[counterexample] adversarial 8x8 input:")
                print(np.array2string(result["adv_input"], precision=4, suppress_small=True))
                break

        sat_result = next((r for r in results if r["exit_code"] == "sat"), None)
        unknown_results = [r for r in results if r["exit_code"] not in {"sat", "unsat"}]

        print("\n=== Verification Summary ===")
        print(f"dataset: sklearn digits")
        print(f"model: external 2-layer CNN exported to ONNX")
        print(f"sample_index: {sample_index}")
        print(f"true_label: {true_label}")
        print(f"epsilon: {args.epsilon:.4f}")
        if sat_result is not None:
            print("final_result: SAT (counterexample found)")
            print(f"counterexample_target: {sat_result['target_label']}")
            print(f"counterexample_prediction: {sat_result['adv_pred']}")
        elif unknown_results:
            print("final_result: UNKNOWN/TIMEOUT")
            print("unresolved_targets:", [r["target_label"] for r in unknown_results])
        else:
            print("final_result: UNSAT for all target labels checked")
            print("interpretation: the sample was locally robust within the tested epsilon")


if __name__ == "__main__":
    main()
