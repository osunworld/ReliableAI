import csv
import os

import cifar10
import fgsm
import mnist
import pgd


EPS_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
NUM_SAMPLES = 1000
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

ATTACK_CONFIGS = [
    {"name": "Targeted FGSM", "method": "FGSM", "targeted": True},
    {"name": "Untargeted FGSM", "method": "FGSM", "targeted": False},
    {"name": "Targeted PGD", "method": "PGD", "targeted": True},
    {"name": "Untargeted PGD", "method": "PGD", "targeted": False},
]

DATASET_CONFIGS = {
    "MNIST": {
        "runner": mnist.run,
        "pgd_k": 40,
        "pgd_eps_step": 0.01,
    },
    "CIFAR-10": {
        "runner": cifar10.run,
        "pgd_k": 10,
        "pgd_eps_step": 0.01,
    },
}


def evaluate_attack_success(model, test_loader, device, attack_config, dataset_config, eps):
    if attack_config["method"] == "FGSM":
        return fgsm.evaluate_fgsm(
            model,
            test_loader,
            device,
            eps=eps,
            targeted=attack_config["targeted"],
            num_samples=NUM_SAMPLES,
        )

    return pgd.evaluate_pgd(
        model,
        test_loader,
        device,
        k=dataset_config["pgd_k"],
        eps=eps,
        eps_step=dataset_config["pgd_eps_step"],
        targeted=attack_config["targeted"],
        num_samples=NUM_SAMPLES,
    )


def build_dataset_table(model, test_loader, device, dataset_name, dataset_config):
    table = {}

    for attack_config in ATTACK_CONFIGS:
        attack_name = attack_config["name"]
        table[attack_name] = {}

        for eps in EPS_VALUES:
            print(f"{dataset_name} | {attack_name} | eps={eps}")
            success_rate = evaluate_attack_success(
                model=model,
                test_loader=test_loader,
                device=device,
                attack_config=attack_config,
                dataset_config=dataset_config,
                eps=eps,
            )
            table[attack_name][eps] = success_rate

    return table


def save_dataset_csv(dataset_name, table):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_name = f"{dataset_name.lower().replace('-', '').replace(' ', '_')}_eps_attack_success.csv"
    output_path = os.path.join(RESULTS_DIR, file_name)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Attack"] + [str(eps) for eps in EPS_VALUES])

        for attack_name in table:
            writer.writerow(
                [attack_name] + [f"{table[attack_name][eps]:.2f}" for eps in EPS_VALUES]
            )

    return output_path


def build_markdown_table(dataset_name, table, dataset_config):
    lines = []
    lines.append(f"## {dataset_name}")
    lines.append("")
    lines.append(
        f"PGD setting: k={dataset_config['pgd_k']}, eps_step={dataset_config['pgd_eps_step']}, "
        f"evaluated on {NUM_SAMPLES} samples"
    )
    lines.append("")

    header = "| Attack | " + " | ".join(str(eps) for eps in EPS_VALUES) + " |"
    separator = "|---|" + "|".join(["---:"] * len(EPS_VALUES)) + "|"
    lines.append(header)
    lines.append(separator)

    for attack_name in table:
        values = " | ".join(f"{table[attack_name][eps]:.2f}" for eps in EPS_VALUES)
        lines.append(f"| {attack_name} | {values} |")

    lines.append("")
    return lines


def save_markdown_summary(all_tables):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "eps_attack_success_summary.md")

    lines = ["# Attack Success Rate by Epsilon", ""]

    for dataset_name, dataset_info in all_tables.items():
        lines.extend(
            build_markdown_table(
                dataset_name=dataset_name,
                table=dataset_info["table"],
                dataset_config=dataset_info["config"],
            )
        )

    with open(output_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(lines))

    return output_path


def print_console_table(dataset_name, table):
    print(f"\n{dataset_name} Attack Success Rate Table (%)")
    header = ["Attack"] + [str(eps) for eps in EPS_VALUES]
    print(" | ".join(header))
    print("-" * 90)

    for attack_name in table:
        row = [attack_name] + [f"{table[attack_name][eps]:.2f}" for eps in EPS_VALUES]
        print(" | ".join(row))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_tables = {}

    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\nLoading {dataset_name} model and test loader")
        model, test_loader, device = dataset_config["runner"]()

        table = build_dataset_table(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )

        csv_path = save_dataset_csv(dataset_name, table)
        print_console_table(dataset_name, table)
        print(f"Saved CSV to {csv_path}")

        all_tables[dataset_name] = {
            "table": table,
            "config": dataset_config,
        }

    markdown_path = save_markdown_summary(all_tables)
    print(f"Saved Markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
