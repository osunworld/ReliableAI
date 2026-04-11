import mnist
import cifar10
import fgsm
import pgd
import visualize


def save_dataset_visualizations(model, test_loader, device, dataset_name, attack_configs):
    print(f"5. Saving {dataset_name} visualizations")

    for config in attack_configs:
        attack_label = f"{'Targeted' if config['targeted'] else 'Untargeted'} {config['attack_method']}"
        print(f"{dataset_name} - {attack_label}")
        visualize.save_visualization(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset_name=dataset_name,
            **config,
        )

def main():
    print("1-1. MNIST Training and Evaluation")
    mnist_model, mnist_test_loader, device = mnist.run()
    print("1-2. CIFAR-10 Training and Evaluation")
    cifar10_model, cifar10_test_loader, device = cifar10.run()

    mnist_attack_configs = [
        {"attack_method": "FGSM", "targeted": True, "eps": 0.1},
        {"attack_method": "FGSM", "targeted": False, "eps": 0.1},
        {"attack_method": "PGD", "targeted": True, "eps": 0.1, "k": 40, "eps_step": 0.01},
        {"attack_method": "PGD", "targeted": False, "eps": 0.1, "k": 40, "eps_step": 0.01},
    ]
    cifar10_attack_configs = [
        {"attack_method": "FGSM", "targeted": True, "eps": 0.03},
        {"attack_method": "FGSM", "targeted": False, "eps": 0.03},
        {"attack_method": "PGD", "targeted": True, "eps": 0.03, "k": 10, "eps_step": 0.01},
        {"attack_method": "PGD", "targeted": False, "eps": 0.03, "k": 10, "eps_step": 0.01},
    ]

    print("2. Targeted FGSM")
    print("MNIST Model")
    fgsm.evaluate_fgsm(mnist_model, mnist_test_loader, device, eps=0.1, targeted=True)
    print("CIFAR-10 Model")
    fgsm.evaluate_fgsm(cifar10_model, cifar10_test_loader, device, eps=0.03, targeted=True)

    print("3. Untargeted FGSM")
    print("MNIST Model")
    fgsm.evaluate_fgsm(mnist_model, mnist_test_loader, device, eps=0.1, targeted=False)
    print("CIFAR-10 Model")
    fgsm.evaluate_fgsm(cifar10_model, cifar10_test_loader, device, eps=0.03, targeted=False)

    print("4-1. Targeted PGD")
    print("MNIST Model")
    pgd.evaluate_pgd(mnist_model, mnist_test_loader, device, k=40, eps=0.1, eps_step=0.01, targeted=True)
    print("CIFAR-10 Model")
    pgd.evaluate_pgd(cifar10_model, cifar10_test_loader, device, k=10, eps=0.03, eps_step=0.01, targeted=True)

    print("4-2. Untargeted PGD")
    print("MNIST Model")
    pgd.evaluate_pgd(mnist_model, mnist_test_loader, device, k=40, eps=0.1, eps_step=0.01, targeted=False)
    print("CIFAR-10 Model")
    pgd.evaluate_pgd(cifar10_model, cifar10_test_loader, device, k=10, eps=0.03, eps_step=0.01, targeted=False)

    save_dataset_visualizations(
        mnist_model,
        mnist_test_loader,
        device,
        dataset_name="MNIST",
        attack_configs=mnist_attack_configs,
    )
    save_dataset_visualizations(
        cifar10_model,
        cifar10_test_loader,
        device,
        dataset_name="CIFAR-10",
        attack_configs=cifar10_attack_configs,
    )

if __name__ == "__main__":
    main()
