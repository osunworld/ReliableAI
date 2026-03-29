import mnist

def main():
    print("1-1. MNIST Training and Evaluation")
    mnist_model, mnist_test_loader, device = mnist.run()


if __name__ == "__main__":
    main()