# Assignment2

## PyTorch DeepXplore Path

The legacy `deepxplore/` clone targets Python 2.7, TensorFlow 1.x, and standalone Keras.
For a modern environment, use the PyTorch-based `CIFAR10_torch/` path instead.

### Dependencies

```bash
pip install torch torchvision transformers pillow safetensors
```

### Model smoke test

```bash
cd ~/Documents/ReliableAI/hw2
python test.py
```

### Differential testing run

```bash
cd ~/Documents/ReliableAI/hw2/CIFAR10_torch
python gen_diff.py light 1.0 0.1 0.01 20 30 0.5
```

This PyTorch path defaults to two CIFAR-10 ResNet50 checkpoints hosted on Hugging Face:

- `jialicheng/cifar10_resnet-50`
- `jialicheng/unlearn-so_cifar10_resnet-50_salun_10_100`
