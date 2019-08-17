import copy
import queue

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
from data import random_rotate
from torch.optim.optimizer import Optimizer
import math


def ssl_train(device, model, dataloaders, num_epochs, num_patches, num_angles, mean, std, learn_prd):
    model = model.to(device)
    optimiser = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 1 / num_angles
    poisson_rate = 0
    for epoch in range(num_epochs):
        if epoch % learn_prd == 0:
            poisson_rate += 1
            dataloaders["train"].dataset.set_poisson_rate(poisson_rate)
        writer.add_scalar("Poisson_Rate", poisson_rate, epoch)

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


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
