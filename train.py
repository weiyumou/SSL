import copy
import math
import queue
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.distributions import Geometric
from torch.utils.tensorboard import SummaryWriter

import utils
from data import random_rotate, DataPrefetcher
from ranger import Ranger
from utils import AverageMeter, reduce_tensor


def ssl_train(device, model, dataloaders, args):
    num_classes = args.num_patches
    model = model.to(device)
    optimiser = Ranger(model.parameters())
    criterion = nn.KLDivLoss(reduction="batchmean")
    gm_dist = Geometric(probs=torch.tensor([1 - 1e-3 ** (1 / num_classes)]))
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 1 / num_classes

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
                    inputs = random_rotate(inputs, args.num_patches, rotations, perms)
                inputs = inputs.to(device)
                labels = torch.exp(gm_dist.log_prob(perms.float())).to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = F.log_softmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                if phase == "train":
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                else:
                    indices = torch.argsort(outputs, dim=1, descending=True)
                    ranks = torch.arange(num_classes, device=device).expand_as(indices)
                    preds = torch.empty_like(indices).scatter_(1, indices, ranks).cpu()
                    running_corrects += torch.sum(preds == perms).item()
                    accuracy_total += perms.numel()

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

        writer.add_scalar("Poisson_Rate", dataloaders["train"].dataset.pdist.rate, epoch)
        if (epoch + 1) % args.learn_prd == 0:
            args.poisson_rate += 1
            dataloaders["train"].dataset.set_poisson_rate(args.poisson_rate)
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, best_val_accuracy


def retrieve_topk_images(device, model, query_img, dataloader, args, k=16):
    model = model.module.to(device).eval()

    writer = SummaryWriter()
    top_image_queue = queue.PriorityQueue(maxsize=k)
    pdist = nn.PairwiseDistance()

    if query_img.dim() == 3:
        query_img = torch.unsqueeze(query_img, dim=0)
    writer.add_images("Query_Image", utils.denormalise(query_img, args.mean, args.std), 0)

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
    writer.add_images("Top_Images", utils.denormalise(top_images, args.mean, args.std), 0)
    writer.close()
    return top_images, list(reversed(top_labels))


def gen_grad_map(device, model, dataloader, args):
    writer = SummaryWriter()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    num_classes = args.num_patches
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
    outputs = outputs.reshape(-1, num_classes)

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


def apex_train(train_loader, model, criterion, optimiser, args, epoch):
    try:
        from apex import amp
    except ImportError:
        args.use_apex = False
        # raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    batch_time = AverageMeter()
    rot_losses = AverageMeter()
    perm_losses = AverageMeter()
    perm_top1 = AverageMeter()
    rot_top1 = AverageMeter()

    model.train()
    end = time.time()
    prefetcher = DataPrefetcher(train_loader, args.num_patches, args.mean, args.std, args.scale)
    for i, (inputs, rot_labels, perm_labels) in enumerate(prefetcher):
        rot_outputs, perm_outputs = model(inputs)
        rot_outputs = rot_outputs.reshape(-1, args.num_angles)
        perm_outputs = perm_outputs.reshape(-1, args.num_patches)

        rot_loss = criterion(rot_outputs, rot_labels)
        perm_loss = criterion(perm_outputs, perm_labels)
        loss = rot_loss + perm_loss

        optimiser.zero_grad()
        if args.use_apex:
            with amp.scale_loss(loss, optimiser) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimiser.step()

        if i % args.print_freq == 0:
            rot_preds = torch.argmax(rot_outputs, dim=1)
            rot_corrects = torch.sum(rot_preds == rot_labels, dim=0, keepdim=True, dtype=torch.float)
            rot_acc = rot_corrects / rot_labels.numel()

            perm_preds = torch.argmax(perm_outputs, dim=1)
            perm_corrects = torch.sum(perm_preds == perm_labels, dim=0, keepdim=True, dtype=torch.float)
            perm_acc = perm_corrects / perm_labels.numel()

            if args.distributed and args.use_apex:
                reduced_perm_loss = reduce_tensor(perm_loss.data, args.world_size)
                reduced_rot_loss = reduce_tensor(rot_loss.data, args.world_size)
                reduced_perm_acc = reduce_tensor(perm_acc.data, args.world_size)
                reduced_rot_acc = reduce_tensor(rot_acc.data, args.world_size)
            else:
                reduced_perm_loss = perm_loss.data
                reduced_perm_acc = perm_acc.data
                reduced_rot_loss = rot_loss.data
                reduced_rot_acc = rot_acc.data

            perm_losses.update(reduced_perm_loss.item(), perm_labels.numel())
            perm_top1.update(reduced_perm_acc.item(), perm_labels.numel())
            rot_losses.update(reduced_rot_loss.item(), rot_labels.numel())
            rot_top1.update(reduced_rot_acc.item(), rot_labels.numel())

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.is_master:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Perm Loss {perm_loss.val:.10f} ({perm_loss.avg:.4f})\t'
                      'Perm Acc {perm_top1.val:.3f} ({perm_top1.avg:.3f})\t'
                      'Rot Loss {rot_loss.val:.10f} ({rot_loss.avg:.4f})\t'
                      'Rot Acc {rot_top1.val:.3f} ({rot_top1.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    perm_loss=perm_losses, perm_top1=perm_top1,
                    rot_loss=rot_losses, rot_top1=rot_top1))

    return rot_losses.avg, rot_top1.avg, perm_losses.avg, perm_top1.avg


def apex_validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    rot_losses = AverageMeter()
    perm_losses = AverageMeter()
    perm_top1 = AverageMeter()
    rot_top1 = AverageMeter()

    model.eval()
    end = time.time()
    prefetcher = DataPrefetcher(val_loader, args.num_patches, args.mean, args.std, args.scale)
    for i, (inputs, rot_labels, perm_labels) in enumerate(prefetcher):
        with torch.no_grad():
            rot_outputs, perm_outputs = model(inputs)
            rot_outputs = rot_outputs.reshape(-1, args.num_angles)
            perm_outputs = perm_outputs.reshape(-1, args.num_patches)

            rot_loss = criterion(rot_outputs, rot_labels)
            perm_loss = criterion(perm_outputs, perm_labels)

        rot_preds = torch.argmax(rot_outputs, dim=1)
        rot_corrects = torch.sum(rot_preds == rot_labels, dim=0, keepdim=True, dtype=torch.float)
        rot_acc = rot_corrects / rot_labels.numel()

        perm_preds = torch.argmax(perm_outputs, dim=1)
        perm_corrects = torch.sum(perm_preds == perm_labels, dim=0, keepdim=True, dtype=torch.float)
        perm_acc = perm_corrects / perm_labels.numel()

        if args.distributed and args.use_apex:
            reduced_perm_loss = reduce_tensor(perm_loss.data, args.world_size)
            reduced_rot_loss = reduce_tensor(rot_loss.data, args.world_size)
            reduced_perm_acc = reduce_tensor(perm_acc.data, args.world_size)
            reduced_rot_acc = reduce_tensor(rot_acc.data, args.world_size)
        else:
            reduced_perm_loss = perm_loss.data
            reduced_perm_acc = perm_acc.data
            reduced_rot_loss = rot_loss.data
            reduced_rot_acc = rot_acc.data

        perm_losses.update(reduced_perm_loss.item(), perm_labels.numel())
        perm_top1.update(reduced_perm_acc.item(), perm_labels.numel())
        rot_losses.update(reduced_rot_loss.item(), rot_labels.numel())
        rot_top1.update(reduced_rot_acc.item(), rot_labels.numel())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.is_master and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Perm Loss {perm_loss.val:.10f} ({perm_loss.avg:.4f})\t'
                  'Perm Acc {perm_top1.val:.3f} ({perm_top1.avg:.3f})\t'
                  'Rot Loss {rot_loss.val:.10f} ({rot_loss.avg:.4f})\t'
                  'Rot Acc {rot_top1.val:.3f} ({rot_top1.avg:.3f})\t'.format(
                i, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, perm_loss=perm_losses, perm_top1=perm_top1,
                rot_loss=rot_losses, rot_top1=rot_top1))

    return rot_losses.avg, rot_top1.avg, perm_losses.avg, perm_top1.avg
