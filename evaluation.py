import torch
import tqdm
import torch.nn as nn


def evaluate_classifier(device, model, eval_loader):
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    corrects = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(eval_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    accuracy = corrects / total
    eval_loss = running_loss / total

    return accuracy, eval_loss
