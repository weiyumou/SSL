import copy

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Subset, DataLoader

import evaluation

from torch.utils.tensorboard import SummaryWriter
import utils
import random
import torchvision.utils as tvutils


def cv_train(device, train_set, fold_indices, modules, num_epochs,
             cv_train_batch_size, cv_val_batch_size, num_classes, test_dataloader):
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

        eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, curr_model, test_dataloader)
        utils.logger.info(f"Val Fold {val_fold}: Test Accuracy = {eval_accuracy}")

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


def ssl_train(device, block, downsampler, dataloader, num_epochs, preprocess):
    block = block.to(device)
    downsampler = downsampler.to(device)
    block_optim = optim.Adam(block.parameters(), lr=5e-3)
    downsampler_optim = optim.Adam(downsampler.parameters())
    criterion = nn.MSELoss()

    block_optim.zero_grad()
    downsampler_optim.zero_grad()

    writer = SummaryWriter()
    # show(tvutils.make_grid(masked_inputs, nrow=4, normalize=True, scale_each=True))
    # show(tvutils.make_grid(masked_labels, nrow=4, normalize=True, scale_each=True))
    conv1_weight = copy.deepcopy(block.layers[0].weight)
    conv1x1_weight = copy.deepcopy(downsampler.conv1x1[0].weight)
    count = 0

    inputs, _ = next(dataloader.__iter__())
    inputs = inputs.to(device)

    for epoch in tqdm.trange(num_epochs):
        random.seed(0)
        # for inputs, _ in tqdm.tqdm(dataloader, desc="SSL Training"):
        # inputs = inputs.to(device)
        with torch.no_grad():
            inputs = preprocess(inputs)

        n, c, h, w = inputs.size()
        masks, mask_h, mask_w = utils.create_masks(n, c, h, w)
        masks = masks.to(device)
        masked_inputs = inputs * (1 - masks)
        masked_labels = inputs[masks == 1].reshape(n, c, mask_h, mask_w)

        fmaps = block(masked_inputs)
        outputs = downsampler(fmaps)
        loss = criterion(outputs, masked_labels)
        loss.backward()
        block_optim.step()
        downsampler_optim.step()

        writer.add_histogram("conv1_grad", block.layers[0].weight.grad, count)
        writer.add_histogram("conv1x1_grad", downsampler.conv1x1[0].weight.grad, count)

        block_optim.zero_grad()
        downsampler_optim.zero_grad()
        utils.logger.info(f"Epoch {epoch}: Loss {loss.item()}")
        # show(tvutils.make_grid(outputs, nrow=4, normalize=True, scale_each=True))
        writer.add_image("Inputs", tvutils.make_grid(masked_inputs, nrow=4, normalize=True, scale_each=True), count)
        writer.add_image("Labels", tvutils.make_grid(masked_labels, nrow=4, normalize=True, scale_each=True), count)
        writer.add_image("Outputs", tvutils.make_grid(outputs, nrow=4, normalize=True, scale_each=True), count)

        conv1_weight_diff = block.layers[0].weight - conv1_weight
        conv1_weight = copy.deepcopy(block.layers[0].weight)
        writer.add_histogram("conv1_diff", conv1_weight_diff, count)

        conv1x1_diff = downsampler.conv1x1[0].weight - conv1x1_weight
        conv1x1_weight = copy.deepcopy(downsampler.conv1x1[0].weight)
        writer.add_histogram("conv1x1_diff", conv1x1_diff, count)
        count += 1

        writer.add_scalar("Loss", loss, count)
    return block, downsampler
