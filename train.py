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


def sl_train(device, dataset, fold_indices, modules, num_epochs,
             train_batch_size, val_batch_size, num_classes, test_dataloader):
    vgg_modules = modules["blocks"] + [modules["classifier"]]
    model = nn.Sequential(*vgg_modules)

    test_accuracy = 0.0
    num_trials, num_examples = fold_indices.shape
    for trial in range(1, num_trials):
        train_indices = fold_indices[trial, :int(0.9 * num_examples)]
        val_indices = fold_indices[trial, int(0.9 * num_examples):]

        curr_model = copy.deepcopy(model)
        dataloaders = {
            "train": DataLoader(Subset(dataset, train_indices), shuffle=False, batch_size=train_batch_size),
            "val": DataLoader(Subset(dataset, val_indices), shuffle=False, batch_size=val_batch_size)}
        curr_model, accuracy = train_classifier(device, curr_model, dataloaders, num_epochs, num_classes)
        utils.logger.info(f"Trial {trial}: Val Accuracy = {accuracy}")

        eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, curr_model, test_dataloader)
        utils.logger.info(f"Trial {trial}: Test Accuracy = {eval_accuracy}")
        test_accuracy += eval_accuracy

    return test_accuracy / num_trials


def train_classifier(device, model, dataloaders, num_epochs, num_classes):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    optimiser.zero_grad()
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 1 / num_classes

    writer = SummaryWriter()
    first_conv = list(list(model.children())[0].layers.children())[0]
    count = 0
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

            conv1_diff = first_conv.weight - conv1_weight
            writer.add_histogram("Conv1_diff", conv1_diff, count)
            count += 1
            break

        epoch_loss = running_loss / total
        utils.logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss}")
        writer.add_scalar("Epoch Loss", epoch_loss, epoch)

        model.eval()
        eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, model, dataloaders["val"])
        utils.logger.info(f"Epoch {epoch}: Val Accuracy = {eval_accuracy}, Val Loss = {eval_loss}")
        writer.add_scalar("Val Accuracy", eval_accuracy, epoch)

        if eval_accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = eval_accuracy

    model.load_state_dict(best_model_wts)
    return model, best_accuracy


def ssl_train(device, block, downsampler, dataloader, num_epochs, preprocess):
    block = block.to(device).train()
    downsampler = downsampler.to(device).train()

    block_optim = optim.Adam(block.parameters(), lr=5e-3)
    downsampler_optim = optim.Adam(downsampler.parameters(), lr=5e-3)
    block_optim.zero_grad()
    downsampler_optim.zero_grad()

    criterion = nn.MSELoss()

    writer = SummaryWriter()
    conv1_weight = copy.deepcopy(block.layers[0].weight)
    conv1x1_weight = copy.deepcopy(downsampler.conv1x1[0].weight)
    count = 0

    for epoch in tqdm.trange(num_epochs):
        if count % 100 == 0:
            random.seed(0)
        for inputs, _ in tqdm.tqdm(dataloader, desc="SSL Training"):
            inputs = inputs.to(device)
            with torch.no_grad():
                preprocess = preprocess.to(device).eval()
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
            if count % 10 == 0:
                break
    return block, downsampler
