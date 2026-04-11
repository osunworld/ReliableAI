import torch
import torch.nn as nn

def fgsm_targeted(model, x, target, eps):
    x.requires_grad_(True)
    outputs = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target)
    model.zero_grad()
    loss.backward()
    data_grad = x.grad.data
    x_adv = x - eps * data_grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def fgsm_untargeted(model, x, label, eps):
    x.requires_grad_(True)
    outputs = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, label)
    model.zero_grad()
    loss.backward()
    data_grad = x.grad.data
    x_adv = x + eps * data_grad.sign() # 정답에 대한 손실을 최대화 하기 위해 기울기를 더함
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def evaluate_fgsm(model, test_loader, device, eps, targeted=True, num_samples=1000):
    model.eval()
    success, total = 0,0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if targeted:
            targets = (labels + 1) % 10 #정답이 아닌 임의의 클래스. 클래스를 지정해줘도 됨
            perturbed_images = fgsm_targeted(model, images, targets, eps)
        else:
            perturbed_images = fgsm_untargeted(model, images, labels, eps)
        
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)

        if targeted:
            success += (predicted == targets).sum().item()
        else:
            success += (predicted != labels).sum().item()

        total += labels.size(0)

        if total >= num_samples:
            break
    
    success_rate = 100 * success / total
    mode_str = "Targeted" if targeted else "Untargeted"
    print(f"{mode_str} FGSM Attack SR (eps={eps}): {success_rate:.2f}%")
    return success_rate
