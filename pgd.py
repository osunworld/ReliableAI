import torch
import torch.nn as nn

def pgd_targeted(model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach().to(x.device)
    criterion = nn.CrossEntropyLoss()

    for _ in range(k):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, target)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data

        with torch.no_grad():
            x_adv = x_adv - eps_step * data_grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach().to(x.device)
    criterion = nn.CrossEntropyLoss()

    for _ in range(k):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, label)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data

        with torch.no_grad():
            x_adv = x_adv + eps_step * data_grad.sign() 
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def evaluate_pgd(model, test_loader, device, k, eps, eps_step, targeted=True, num_samples=1000):
    model.eval()
    success, total = 0,0    

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if targeted:
            targets = (labels + 1) % 10
            perturbed_images = pgd_targeted(model, images, targets, k, eps, eps_step)
        else:
            perturbed_images = pgd_untargeted(model, images, labels, k, eps, eps_step)
        
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
    print(f"{mode_str} PGD Attack SR (eps={eps}, k={k}, step={eps_step}): {success_rate:.2f}%")
    return success_rate