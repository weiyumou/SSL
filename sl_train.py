import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import utils


def train_classifier(device, model, dataloaders, args):
    model = model.to(device)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            utils.logger.info(f"Param to update: {name}")

    optimiser = optim.Adam(params_to_update)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 1 / args.num_classes
    writer = SummaryWriter()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        total = 0

        model.train()
        for inputs, labels in tqdm.tqdm(dataloaders["train"], desc="SL Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * labels.numel()
            total += labels.numel()

        epoch_loss = running_loss / total
        utils.logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss}")

        model.eval()
        eval_accuracy, eval_loss = evaluate_classifier(device, model, dataloaders["val"])
        utils.logger.info(f"Epoch {epoch}: Val Accuracy = {eval_accuracy}, Val Loss = {eval_loss}")

        if eval_accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = eval_accuracy

        writer.add_scalars("Loss", {"Train": epoch_loss, "Val": eval_loss}, epoch)
        writer.add_scalar("Accuracy/Val", eval_accuracy, epoch)

    model.load_state_dict(best_model_wts)
    writer.close()
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
        running_loss += loss.item() * labels.numel()
        total += labels.numel()
    accuracy = corrects / total
    eval_loss = running_loss / total

    return accuracy, eval_loss


def stl_sl_train(device, model, train_dataset, fold_indices, dataloaders, args):
    test_accuracy = 0.0
    num_folds, num_examples = fold_indices.shape
    for fold in range(num_folds):
        train_indices = fold_indices[fold, :int(0.9 * num_examples)]
        val_indices = fold_indices[fold, int(0.9 * num_examples):]

        curr_model = copy.deepcopy(model)
        dataloaders["train"] = DataLoader(Subset(train_dataset, train_indices), shuffle=True,
                                          batch_size=args.train_batch_size, pin_memory=True)
        dataloaders["val"] = DataLoader(Subset(train_dataset, val_indices), shuffle=False,
                                        batch_size=args.val_batch_size, pin_memory=True)
        curr_model, accuracy = train_classifier(device, curr_model, dataloaders, args)
        utils.logger.info(f"Trial {fold}: Val Accuracy = {accuracy}")

        eval_accuracy, eval_loss = evaluate_classifier(device, curr_model, dataloaders["test"])
        utils.logger.info(f"Trial {fold}: Test Accuracy = {eval_accuracy}")
        test_accuracy += eval_accuracy

    return test_accuracy / num_folds


def stl_get_train_folds(fold_file, num_folds=10):
    fold_indices = np.fromfile(fold_file, dtype=int, sep=" ")
    return fold_indices.reshape(num_folds, -1)
