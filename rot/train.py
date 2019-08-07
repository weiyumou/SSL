import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils


def random_rotate(images, num_patches, num_angles, rotations):
    n, c, img_h, img_w = images.size()

    labels = []
    patch_size = int(img_h / math.sqrt(num_patches))
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = patches.reshape(n, c, patch_size, patch_size, num_patches)
    for img_idx in range(n):
        for patch_idx in range(num_patches):
            patches[img_idx, :, :, :, patch_idx] = torch.rot90(patches[img_idx, :, :, :, patch_idx],
                                                               rotations[img_idx, patch_idx].item(), [1, 2])
        label = torch.zeros(num_patches, num_angles).scatter_(1, torch.unsqueeze(rotations[img_idx, :], 1), 1)
        perm = torch.randperm(num_patches, dtype=torch.long)
        patches[img_idx] = patches[img_idx, :, :, :, perm]
        label = label[perm]
        labels.append(label)

    patches = patches.reshape(n, -1, num_patches)
    images = F.fold(patches, output_size=img_h, kernel_size=patch_size, stride=patch_size)
    labels = torch.stack(labels)
    return images, labels


def ssl_train(device, model, dataloaders, num_epochs, num_patches, num_angles):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    optimiser.zero_grad()
    criterion = nn.MultiLabelSoftMarginLoss()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 1 / (num_patches * num_angles)

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.0
            loss_total = 0
            running_corrects = 0
            accuracy_total = 0

            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, rotations in tqdm.tqdm(dataloaders[phase], desc=f"SSL {phase}"):
                with torch.no_grad():
                    inputs, labels = random_rotate(inputs, num_patches, num_angles, rotations)
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.reshape(labels.size(0), -1))

                if phase == "train":
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                else:
                    outputs = outputs.reshape(-1, num_patches, num_angles)
                    preds = torch.argmax(outputs, dim=2)
                    labels = torch.argmax(labels, dim=2)
                    running_corrects += torch.sum(preds == labels).item()
                    accuracy_total += labels.numel()

                running_loss += loss.item() * inputs.size(0)
                loss_total += inputs.size(0)

            epoch_loss = running_loss / loss_total
            utils.logger.info(f"Epoch {epoch}: {phase} loss = {epoch_loss}")
            writer.add_scalar(f"{phase}_loss", epoch_loss, epoch)

            if phase == "val":
                epoch_accuracy = running_corrects / accuracy_total
                if epoch_accuracy > best_val_accuracy:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_accuracy = epoch_accuracy
                utils.logger.info(f"Epoch {epoch}: {phase} accuracy = {epoch_accuracy}")
                writer.add_scalar(f"{phase}_accuracy", epoch_accuracy, epoch)

    model.load_state_dict(best_model_wts)
    return model, best_val_accuracy
