import copy
import math
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils


def random_rotate(images, num_patches, rotations, perms=None):
    n, c, img_h, img_w = images.size()

    patch_size = int(img_h / math.sqrt(num_patches))
    patches = F.unfold(images, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    patches = patches.reshape(n, c, patch_size, patch_size, num_patches)
    for img_idx in range(n):
        for patch_idx in range(num_patches):
            patches[img_idx, :, :, :, patch_idx] = torch.rot90(patches[img_idx, :, :, :, patch_idx],
                                                               rotations[img_idx, patch_idx].item(), [1, 2])
        if perms is not None:
            patches[img_idx] = patches[img_idx, :, :, :, perms[img_idx]]
            rotations[img_idx] = rotations[img_idx, perms[img_idx]]

    patches = patches.reshape(n, -1, num_patches)
    images = F.fold(patches, output_size=img_h, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    return images, torch.flatten(rotations)


def ssl_train(device, model, dataloaders, num_epochs, num_patches, num_angles, mean, std):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 1 / num_angles

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

            for inputs, rotations, perms in tqdm.tqdm(dataloaders[phase], desc=f"SSL {phase}"):
                # writer.add_images("Raw Inputs", utils.denormalise(inputs, mean, std), epoch)
                with torch.no_grad():
                    inputs, labels = random_rotate(inputs, num_patches, rotations, perms)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # writer.add_images("Inputs", utils.denormalise(inputs, mean, std), epoch)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = outputs.reshape(-1, num_angles)
                    loss = criterion(outputs, labels)

                if phase == "train":
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                else:
                    preds = torch.argmax(outputs, dim=1)
                    running_corrects += torch.sum(preds == labels).item()
                    accuracy_total += labels.numel()

                running_loss += loss.item() * labels.numel()
                loss_total += labels.numel()

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
    writer.close()
    return model, best_val_accuracy


def retrieve_topk_images(device, model, query_img, dataloader, mean, std, k=16):
    model = model.to(device).eval()

    writer = SummaryWriter()
    top_image_queue = queue.PriorityQueue(maxsize=k)
    pdist = nn.PairwiseDistance()

    if query_img.dim() == 3:
        query_img = torch.unsqueeze(query_img, dim=0)
    writer.add_images("Query Image", utils.denormalise(query_img, mean, std), 0)

    with torch.no_grad():
        query_vec = model.backend(query_img.to(device))
        for images, labels in dataloader:
            fvecs = model.backend(images.to(device))
            dists = pdist(query_vec, fvecs).tolist()
            for idx, dist in enumerate(dists):
                if top_image_queue.full():
                    last_item = top_image_queue.get()
                    if -dist > last_item[0]:
                        top_image_queue.put((-dist, (images[idx], labels[idx].item())))
                else:
                    top_image_queue.put((-dist, (images[idx], labels[idx].item())))

    top_images, top_labels = [], []
    while not top_image_queue.empty():
        _, (image, label) = top_image_queue.get()
        top_images.append(image)
        top_labels.append(label)
    top_images = torch.cat(tuple(reversed(top_images)), dim=0)
    top_labels = torch.cat(tuple(reversed(top_labels)), dim=0)
    writer.add_images("Top Images", utils.denormalise(top_images, mean, std), 0)
    writer.close()
