import mnist
import cifar10

def main():
    print("1-1. MNIST Training and Evaluation")
    mnist_model, mnist_test_loader, device = mnist.run()
    print("1-2. CIFAR-10 Training and Evaluation")
    cifar10_model, cifar10_test_loader, device = cifar10.run()


if __name__ == "__main__":
    main()