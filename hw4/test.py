#!/usr/bin/env python3
"""Prepare and run the HW4 alpha-beta-CROWN experiment.

This file intentionally contains both the executable runner and the custom
model/dataloader functions used by alpha-beta-CROWN. The YAML config added in a
later commit can load these functions with:

  Customized("test.py", "digits_cnn")
  Customized("test.py", "digits_test_data", sample_index=348, seed=7)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "digits_linf.yaml"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "digits_cnn.pth"
RESULTS_DIR = ROOT / "results"
DEFAULT_INSTALL_CONFIG = Path("exp_configs") / "tutorial_examples" / "custom_box_data_example.yaml"


class SmallDigitsCNN(nn.Module):
    """Small 2-layer CNN for 8x8 grayscale sklearn digits images."""

    def __init__(self) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def digits_cnn() -> nn.Module:
    """Return the architecture used by alpha-beta-CROWN to load the checkpoint."""

    return SmallDigitsCNN()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sklearn-digits CNN and verify it with alpha-beta-CROWN."
    )
    parser.add_argument(
        "--abcrown-root",
        default=os.environ.get("ABCROWN_ROOT", str(ROOT / "alpha-beta-CROWN")),
        help=(
            "Path to the alpha-beta-CROWN repository or its complete_verifier directory. "
            "Can also be provided through ABCROWN_ROOT."
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config passed to abcrown.py.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-index", type=int, default=348)
    parser.add_argument("--epsilon", type=float, default=None, help="Override specification.epsilon.")
    parser.add_argument("--timeout", type=float, default=None, help="Override bab.timeout.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if the checkpoint exists.")
    parser.add_argument(
        "--check-install",
        action="store_true",
        help="Run a small built-in alpha-beta-CROWN example before the HW4 experiment.",
    )
    parser.add_argument(
        "--check-install-only",
        action="store_true",
        help="Only run a small built-in alpha-beta-CROWN example and then exit.",
    )
    parser.add_argument(
        "--install-config",
        default=str(DEFAULT_INSTALL_CONFIG),
        help="Example config path, relative to complete_verifier, used for installation checking.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only train/save the model and metadata; do not launch alpha-beta-CROWN.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def prepare_data(seed: int):
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    x = np.expand_dims(x, axis=1)
    y = digits.target.astype(np.int64)
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)


def _digits_test_split(seed: int):
    _, x_test, _, y_test = prepare_data(seed)
    return x_test, y_test


def digits_test_data(spec, sample_index: int = 348, seed: int = 7):
    """Custom alpha-beta-CROWN dataloader for one normalized digits sample."""

    eps = spec["epsilon"]
    if eps is None:
        raise ValueError("The digits robustness config requires specification.epsilon.")

    x_test, y_test = _digits_test_split(seed)
    if not 0 <= sample_index < len(x_test):
        raise ValueError(
            f"sample_index must be between 0 and {len(x_test) - 1}, got {sample_index}."
        )

    x = torch.from_numpy(x_test[sample_index : sample_index + 1]).float()
    labels = torch.tensor([int(y_test[sample_index])], dtype=torch.long)

    # Inputs are already normalized to [0, 1], so these bounds clip the Linf ball
    # to valid pixel values without additional mean/std conversion.
    data_max = torch.ones(1, 1, 1, 1)
    data_min = torch.zeros(1, 1, 1, 1)
    eps_tensor = torch.tensor(float(eps)).reshape(1, 1, 1, 1)
    return x, labels, data_max, data_min, eps_tensor


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
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
            print(f"[train] epoch={epoch:02d} loss={total_loss / total_examples:.4f}")


def evaluate_model(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test)).cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    accuracy = float((predictions == y_test).mean())
    sorted_logits = np.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    return logits, predictions, margins, accuracy


def normalize_abcrown_root(path: Path) -> Path:
    if path.name == "complete_verifier":
        complete_verifier = path
    else:
        complete_verifier = path / "complete_verifier"
    abcrown_py = complete_verifier / "abcrown.py"
    if not abcrown_py.exists():
        raise SystemExit(
            "alpha-beta-CROWN was not found. Pass --abcrown-root /path/to/alpha-beta-CROWN "
            "or set ABCROWN_ROOT. Expected abcrown.py at "
            f"{abcrown_py}."
        )
    return complete_verifier


def run_install_check(args: argparse.Namespace) -> int:
    complete_verifier = normalize_abcrown_root(Path(args.abcrown_root).expanduser().resolve())
    install_config = Path(args.install_config)
    if not install_config.is_absolute():
        install_config = complete_verifier / install_config
    if not install_config.exists():
        raise SystemExit(f"alpha-beta-CROWN example config not found: {install_config}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "abcrown.py",
        "--config",
        str(install_config),
        "--device",
        args.device,
    ]
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])

    print("[install-check] running built-in alpha-beta-CROWN example:")
    print(" ".join(command))
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=complete_verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start

    log_path = RESULTS_DIR / "abcrown_install_check.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    summary = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "example_config": str(install_config),
        "log": str(log_path),
        "notes": [
            "Checked alpha-beta-CROWN installation with a provided tutorial example.",
            "The custom_box_data_example uses a tiny toy model and does not require downloading image datasets.",
        ],
    }
    summary_path = RESULTS_DIR / "abcrown_install_check.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[install-check] returncode={completed.returncode} elapsed={elapsed:.2f}s")
    print(f"[install-check] wrote log: {log_path}")
    print(f"[install-check] wrote summary: {summary_path}")
    return completed.returncode


def save_checkpoint_and_metadata(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    x_train, x_test, y_train, y_test = prepare_data(args.seed)

    model = SmallDigitsCNN()
    if DEFAULT_CHECKPOINT.exists() and not args.retrain:
        print(f"[setup] loading existing checkpoint: {DEFAULT_CHECKPOINT}")
        model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True))
    else:
        print("[setup] training external sklearn-digits CNN")
        train_model(model, x_train, y_train, args.epochs, args.batch_size, args.learning_rate)
        DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), DEFAULT_CHECKPOINT)
        print(f"[setup] saved checkpoint: {DEFAULT_CHECKPOINT}")

    _, predictions, margins, accuracy = evaluate_model(model, x_test, y_test)
    if not 0 <= args.sample_index < len(x_test):
        raise SystemExit(f"--sample-index must be between 0 and {len(x_test) - 1}")

    sample_true = int(y_test[args.sample_index])
    sample_pred = int(predictions[args.sample_index])
    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "test_accuracy": accuracy,
        "sample_index": args.sample_index,
        "sample_true_label": sample_true,
        "sample_predicted_label": sample_pred,
        "sample_margin": float(margins[args.sample_index]),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "config": str(Path(args.config).resolve()),
    }
    metadata_path = RESULTS_DIR / "digits_setup.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[eval] test_accuracy={accuracy:.4f}")
    print(
        "[sample] "
        f"index={args.sample_index} true={sample_true} pred={sample_pred} "
        f"margin={metadata['sample_margin']:.4f}"
    )
    print(f"[setup] wrote metadata: {metadata_path}")
    if sample_true != sample_pred:
        print("[warning] selected sample is misclassified, so robustness is not meaningful for this sample.")
    return metadata


def run_abcrown(args: argparse.Namespace) -> int:
    complete_verifier = normalize_abcrown_root(Path(args.abcrown_root).expanduser().resolve())
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(
            f"Config file not found: {config_path}. Add the YAML config before running verification."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "abcrown.py",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--results_file",
        str(RESULTS_DIR / "abcrown_digits_linf.pkl"),
    ]
    if args.epsilon is not None:
        command.extend(["--epsilon", str(args.epsilon)])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])

    print("[abcrown] running:")
    print(" ".join(command))
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=complete_verifier,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start

    log_path = RESULTS_DIR / "abcrown_stdout.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    summary = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "abcrown_stdout": str(log_path),
    }
    (RESULTS_DIR / "abcrown_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[abcrown] returncode={completed.returncode} elapsed={elapsed:.2f}s")
    print(f"[abcrown] wrote log: {log_path}")
    result_file = RESULTS_DIR / "abcrown_digits_linf.pkl"
    if result_file.exists():
        print(f"[abcrown] result file: {result_file}")
        with result_file.open("rb") as f:
            result_payload = pickle.load(f)
        printable_payload = {
            "summary": {key: value for key, value in result_payload.get("summary", {}).items()},
            "results": result_payload.get("results", []),
            "bab_ret": result_payload.get("bab_ret", []),
        }
        summary_path = RESULTS_DIR / "abcrown_result_summary.json"
        summary_path.write_text(json.dumps(printable_payload, indent=2), encoding="utf-8")
        print(json.dumps(printable_payload, indent=2))
        print(f"[abcrown] wrote summary: {summary_path}")
    else:
        print("[abcrown] result file was not produced; inspect the stdout log.")
    return completed.returncode


def main() -> None:
    args = parse_args()
    if args.check_install or args.check_install_only:
        install_returncode = run_install_check(args)
        if args.check_install_only or install_returncode != 0:
            raise SystemExit(install_returncode)

    save_checkpoint_and_metadata(args)
    if args.prepare_only:
        print("[done] prepare-only mode; alpha-beta-CROWN was not launched.")
        return
    raise SystemExit(run_abcrown(args))


if __name__ == "__main__":
    main()
