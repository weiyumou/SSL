import copy

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Subset, DataLoader

import evaluation

from torch.utils.tensorboard import SummaryWriter
import utils


def cv_train(device, train_set, fold_indices, modules, num_epochs,
             cv_train_batch_size, cv_val_batch_size, num_classes):
    row_indices = set(range(fold_indices.shape[0]))

    vgg_modules = modules["blocks"] + [modules["classifier"]]
    model = nn.Sequential(*vgg_modules)

    best_model = model
    best_accuracy = 1 / num_classes
    for val_fold in row_indices:
        train_folds = list(row_indices - {val_fold})
        train_indices = fold_indices[train_folds, :].reshape(-1)
        val_indices = fold_indices[val_fold]

        curr_model = copy.deepcopy(model)
        dataloaders = {
            "train": DataLoader(Subset(train_set, train_indices), shuffle=True, batch_size=cv_train_batch_size),
            "val": DataLoader(Subset(train_set, val_indices), shuffle=False, batch_size=cv_val_batch_size)}
        curr_model, accuracy = train_classifier(device, curr_model, dataloaders, num_epochs, num_classes)
        utils.logger.info(f"Val Fold {val_fold}: Best Accuracy = {accuracy}")
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(curr_model)
            best_accuracy = accuracy

    return best_model, best_accuracy


def train_classifier(device, model, dataloaders, num_epochs, num_classes):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimiser.zero_grad()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 1 / num_classes

    writer = SummaryWriter()
    first_conv = list(list(model.children())[0].layers.children())[0]
    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0

        model.train()
        for inputs, labels in tqdm.tqdm(dataloaders["train"], desc="Training"):
            conv1_weight = copy.deepcopy(first_conv.weight)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            conv1_diff = torch.mean(first_conv.weight - conv1_weight)
            writer.add_histogram("Conv1_diff", conv1_diff, global_step)
            global_step += 1

        epoch_loss = running_loss / total
        utils.logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss}")

        model.eval()
        eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, model, dataloaders["val"])
        utils.logger.info(f"Epoch {epoch}: Val Accuracy = {eval_accuracy}, Val Loss = {eval_loss}")

        if eval_accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = eval_accuracy

    model.load_state_dict(best_model_wts)
    return model, best_accuracy
