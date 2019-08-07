import copy

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Subset, DataLoader

import sl_train
import utils

from torch.utils.tensorboard import SummaryWriter
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
        curr_model, accuracy = sl_train.train_classifier(device, curr_model, dataloaders, num_epochs, num_classes)
        utils.logger.info(f"Trial {trial}: Val Accuracy = {accuracy}")

        eval_accuracy, eval_loss = sl_train.evaluate_classifier(device, curr_model, test_dataloader)
        utils.logger.info(f"Trial {trial}: Test Accuracy = {eval_accuracy}")
        test_accuracy += eval_accuracy

    return test_accuracy / num_trials


def ssl_train(device, block, downsampler, dataloader, num_epochs, preprocess):
    block = block.to(device).train()
    downsampler = downsampler.to(device).train()
    preprocess = preprocess.to(device).eval()

    block_optim = optim.Adam(block.parameters(), lr=7e-4)

    downsampler_optim = optim.Adam(downsampler.parameters(), lr=1e-3)
    block_optim.zero_grad()
    downsampler_optim.zero_grad()

    criterion = nn.MSELoss()

    writer = SummaryWriter()
    count = 0

    for epoch in tqdm.trange(num_epochs):
        total = 0
        running_loss = 0.0
        # if count % 10 == 0:
        #     random.seed(0)
        for inputs, _ in tqdm.tqdm(dataloader, desc="SSL Training"):
            block_params = dict(copy.deepcopy(list(block.named_parameters())))
            downsampler_params = dict(copy.deepcopy(list(downsampler.named_parameters())))

            # for layer, param in block.named_parameters():
            #     writer.add_histogram(f"block_{layer}", param.data, count)
            #
            # for layer, param in downsampler.named_parameters():
            #     writer.add_histogram(f"downsampler_{layer}", param.data, count)

            inputs = inputs.to(device)
            with torch.no_grad():
                inputs = preprocess(inputs)

            n, c, h, w = inputs.size()
            masks, mask_h, mask_w = create_masks(n, c, h, w)
            masks = masks.to(device)
            masked_inputs = inputs * (1 - masks)
            masked_labels = inputs[masks == 1].reshape(n, c, mask_h, mask_w)

            block_activations = block(masked_inputs)
            fmaps = block_activations[-1]
            downsampler_activations = downsampler(fmaps)
            outputs = downsampler_activations[-1]
            loss = criterion(outputs, masked_labels)
            loss.backward()
            block_optim.step()
            downsampler_optim.step()

            block_optim.zero_grad()
            downsampler_optim.zero_grad()
            utils.logger.info(f"Epoch {epoch}: Loss {loss.item()}")

            writer.add_image("Inputs", tvutils.make_grid(masked_inputs, nrow=4, normalize=True, scale_each=True), count)
            writer.add_image("Labels", tvutils.make_grid(masked_labels, nrow=4, normalize=True, scale_each=True), count)
            writer.add_image("Outputs", tvutils.make_grid(outputs, nrow=4, normalize=True, scale_each=True), count)

            for index, (layer, param) in enumerate(block.named_parameters()):
                weight_update = param.data - block_params[layer].data
                update_ratio = torch.mean(torch.abs(weight_update)) / torch.mean(torch.abs(block_params[layer].data))
                # update_ratio = torch.norm(weight_update.reshape(-1)) / torch.norm(block_params[layer].data.reshape(-1))
                writer.add_scalar(f"block_{layer}_update_ratio", update_ratio, count)

            # for index, (num, layer) in enumerate(block.layers.named_children()):
            #     writer.add_histogram(f"block_{layer.__class__.__name__}_{num}", block_activations[index], count)

            for index, (layer, param) in enumerate(downsampler.named_parameters()):
                weight_update = param.data - downsampler_params[layer].data
                update_ratio = torch.mean(torch.abs(weight_update)) / torch.mean(torch.abs(downsampler_params[layer].data))
                # update_ratio = torch.norm(weight_update.reshape(-1)) / torch.norm(
                #     downsampler_params[layer].data.reshape(-1))
                writer.add_scalar(f"downsampler_{layer}_update_ratio", update_ratio, count)

            # for index, (num, layer) in enumerate(downsampler.conv1x1.named_children()):
            #     writer.add_histogram(f"downsampler_{layer.__class__.__name__}_{num}", downsampler_activations[index],
            #                          count)
            # writer.add_histogram(f"downsampler_{downsampler.fc.__class__.__name__}", downsampler_activations[-1], count)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            count += 1
            if count % 10 == 0:
                break

        epoch_loss = running_loss / total
        writer.add_scalar("Loss", epoch_loss, epoch)
    return block, downsampler


def create_masks(n, c, h, w):
    mask_h, mask_w = h // 2, w // 2
    masks = []
    for idx in range(n):
        left_h = random.randint(0, mask_h)
        left_w = random.randint(0, mask_w)
        mask = torch.zeros(h, w)
        mask[left_h: left_h + mask_h, left_w: left_w + mask_w] = 1
        mask = mask.expand(c, h, w)
        masks.append(mask)
    masks = torch.stack(masks)
    return masks, mask_h, mask_w
