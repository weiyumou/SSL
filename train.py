import copy
import math
import queue

import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
from data import random_rotate, DataPrefetcher
from ranger import Ranger
from utils import AverageMeter, reduce_tensor


def ssl_train(device, model, dataloaders, args):
    model = model.to(device)
    optimiser = Ranger(model.parameters())
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 1 / args.num_angles

    for epoch in range(args.ssl_num_epochs):
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
                with torch.no_grad():
                    inputs, labels = random_rotate(inputs, args.num_patches, rotations, perms)
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = outputs.reshape(-1, args.num_angles)
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

        writer.add_scalar("Poisson_Rate", dataloaders["train"].dataset.pdist.rate, epoch)
        if (epoch + 1) % args.learn_prd == 0:
            args.poisson_rate += 1
            dataloaders["train"].dataset.set_poisson_rate(args.poisson_rate)
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
    writer.add_images("Query_Image", utils.denormalise(query_img, mean, std), 0)

    with torch.no_grad():
        query_vec = torch.flatten(model.backend(query_img.to(device)), start_dim=1)
        for images, labels in dataloader:
            fvecs = torch.flatten(model.backend(images.to(device)), start_dim=1)
            dists = pdist(query_vec, fvecs).tolist()
            for idx, dist in enumerate(dists):
                if top_image_queue.full():
                    last_item = top_image_queue.get()
                    if -dist > last_item[0]:
                        top_image_queue.put((-dist, (images[idx], labels[idx].item())))
                    else:
                        top_image_queue.put(last_item)
                else:
                    top_image_queue.put((-dist, (images[idx], labels[idx].item())))

    top_images, top_labels = [], []
    while not top_image_queue.empty():
        _, (image, label) = top_image_queue.get()
        top_images.append(image.detach().cpu())
        top_labels.append(label)
    top_images = torch.stack(tuple(reversed(top_images)), dim=0)
    writer.add_images("Top_Images", utils.denormalise(top_images, mean, std), 0)
    writer.close()
    return top_images, list(reversed(top_labels))


def gen_grad_map(device, model, dataloader, args):
    writer = SummaryWriter()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    data_iter = iter(dataloader)
    inputs, rotations, perms = next(data_iter)
    with torch.no_grad():
        inputs, labels = random_rotate(inputs, args.num_patches, rotations, perms)

    writer.add_images("Input", utils.denormalise(inputs, args.mean, args.std), 0)
    n, c, h, w = inputs.size()
    inputs = inputs.to(device).requires_grad_()
    labels = labels.to(device)

    outputs = model(inputs)
    outputs = outputs.reshape(-1, args.num_angles)

    for image_idx in range(n):
        image_grads = []
        for num in range(args.num_patches):
            patch_idx = num + args.num_patches * image_idx

            loss = criterion(outputs[patch_idx].unsqueeze(dim=0), labels[patch_idx].unsqueeze(dim=0))
            model.zero_grad()
            loss.backward(retain_graph=True)

            grad = torch.abs(inputs.grad[image_idx, -1])
            min_grad = torch.min(grad)
            max_grad = torch.max(grad)
            image_grads.append(grad.sub(min_grad).div(max_grad - min_grad))
            inputs.grad = torch.zeros_like(inputs.grad)

        image_grads = torch.stack(image_grads, dim=2).reshape(1, -1, args.num_patches)
        image_grads = torch.nn.functional.fold(image_grads, output_size=h * int(math.sqrt(args.num_patches)),
                                               kernel_size=h,
                                               stride=h)
        writer.add_images(f"Grad_Map/{image_idx}", image_grads, 0)

    writer.close()


def apex_train(train_loader, model, criterion, optimiser, args):
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    prefetcher = DataPrefetcher(train_loader, args.num_patches, args.mean, args.std, args.scale)
    for i, (inputs, labels) in tqdm.tqdm(enumerate(prefetcher), desc="SSL Training", total=len(train_loader)):
        outputs = model(inputs)
        outputs = outputs.reshape(-1, args.num_angles)
        loss = criterion(outputs, labels)

        optimiser.zero_grad()
        with amp.scale_loss(loss, optimiser) as scaled_loss:
            scaled_loss.backward()
        optimiser.step()

        if i % args.print_freq == 0:
            preds = torch.argmax(outputs, dim=1)
            corrects = torch.sum(preds == labels, dim=0, keepdim=True, dtype=torch.float)
            acc = corrects / labels.numel()
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_acc = reduce_tensor(acc.data, args.world_size)
            else:
                reduced_loss = loss.data
                reduced_acc = acc.data
            losses.update(reduced_loss.item(), labels.numel())
            top1.update(reduced_acc.item(), labels.numel())

    return losses.avg, top1.avg


def apex_validate(val_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    prefetcher = DataPrefetcher(val_loader, args.num_patches, args.mean, args.std, args.scale)
    for inputs, labels in tqdm.tqdm(prefetcher, desc="SSL Evaluating", total=len(val_loader)):
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.reshape(-1, args.num_angles)
            loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        corrects = torch.sum(preds == labels, dim=0, keepdim=True, dtype=torch.float)
        acc = corrects / labels.numel()
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_acc = reduce_tensor(acc.data, args.world_size)
        else:
            reduced_loss = loss.data
            reduced_acc = acc.data
        losses.update(reduced_loss.item(), labels.numel())
        top1.update(reduced_acc.item(), labels.numel())

    return losses.avg, top1.avg
