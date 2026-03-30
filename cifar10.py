import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super(NormalizedModel, self).__init__()
        self.base_model = base_model
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1,))

    def forward(self, x):
        x_norm = (x-self.mean) / self.std
        return self.base_model(x_norm)

def get_pt_model(device):
        print("Loading PT CIFAR-10 model")
        base_model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
        model = NormalizedModel(base_model).to(device)
        model.eval()
        return model

def evaluate(model, test_loader, device):
    model.eval()
    correct, total=0,0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100*correct/total
    print(f"Clean Test Accuracy: {acc:.2f}%")
    return acc

def run():
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    print(f"device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root='./data/CIFAR_10', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = get_pt_model(device)
    acc = evaluate(model, test_loader, device)

    if acc >= 80.0:
        print("CIFAR-10 test passed!")
    else:
        print("CIFAR-10 test failed!")

    return model, test_loader, device