import mnist
import cifar10
import fgsm

def main():
    print("1-1. MNIST Training and Evaluation")
    mnist_model, mnist_test_loader, device = mnist.run()
    print("1-2. CIFAR-10 Training and Evaluation")
    cifar10_model, cifar10_test_loader, device = cifar10.run()

    print("2. Targeted FGSM")
    print("MNIST Model")
    fgsm.evaluate_targeted_fgsm(mnist_model, mnist_test_loader, device, eps=0.3)
    print("CIFAR-10 Model")
    fgsm.evaluate_targeted_fgsm(cifar10_model, cifar10_test_loader, device, eps=0.05)

if __name__ == "__main__":
    main()