import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Subset

import utils


def train_classifier(device, model, dataloaders, num_epochs, num_classes):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    optimiser.zero_grad()
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 1 / num_classes

    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0

        model.train()
        for inputs, labels in tqdm.tqdm(dataloaders["train"], desc="SL Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        utils.logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss}")

        model.eval()
        eval_accuracy, eval_loss = evaluate_classifier(device, model, dataloaders["val"])
        utils.logger.info(f"Epoch {epoch}: Val Accuracy = {eval_accuracy}, Val Loss = {eval_loss}")

        if eval_accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = eval_accuracy

    model.load_state_dict(best_model_wts)
    return model, best_accuracy


def evaluate_classifier(device, model, eval_loader):
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    corrects = 0
    total = 0
    running_loss = 0.0

    for inputs, labels in tqdm.tqdm(eval_loader, desc="SL Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        corrects += torch.sum(preds == labels).item()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    accuracy = corrects / total
    eval_loss = running_loss / total

    return accuracy, eval_loss


def stl_sl_train(device, model, train_dataset, fold_indices, num_epochs,
                 train_batch_size, val_batch_size, num_classes, test_dataloader):
    test_accuracy = 0.0
    num_trials, num_examples = fold_indices.shape
    for trial in range(num_trials):
        train_indices = fold_indices[trial, :int(0.9 * num_examples)]
        val_indices = fold_indices[trial, int(0.9 * num_examples):]

        curr_model = copy.deepcopy(model)
        dataloaders = {
            "train": DataLoader(Subset(train_dataset, train_indices), shuffle=True, batch_size=train_batch_size),
            "val": DataLoader(Subset(train_dataset, val_indices), shuffle=False, batch_size=val_batch_size)
        }
        curr_model, accuracy = train_classifier(device, curr_model, dataloaders, num_epochs, num_classes)
        utils.logger.info(f"Trial {trial}: Val Accuracy = {accuracy}")

        eval_accuracy, eval_loss = evaluate_classifier(device, curr_model, test_dataloader)
        utils.logger.info(f"Trial {trial}: Test Accuracy = {eval_accuracy}")
        test_accuracy += eval_accuracy

    return test_accuracy / num_trials


def stl_get_train_folds(fold_file, num_folds=10):
    fold_indices = np.fromfile(fold_file, dtype=int, sep=" ")
    return fold_indices.reshape(num_folds, -1)
