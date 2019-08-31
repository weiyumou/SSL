import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributed as dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def show(img, mean, std):
    plt.figure()
    plt.xticks([])
    plt.yticks([])

    mean = np.array(mean)
    std = np.array(std)

    num_channels, *_ = img.size()
    npimg = img.detach().cpu().numpy()
    npimg = denormalise(npimg, mean, std, batched=False)

    if num_channels == 1:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def show_mask(mask):
    plt.figure()
    np_mask = mask.detach().cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np_mask, cmap='gray')


def calc_images_mean_std(dataloader, num_channels=3):
    first_moment = torch.zeros(num_channels)
    sec_moment = torch.zeros(num_channels)
    count = 1
    for inputs, *_ in dataloader:
        n, _, h, w = inputs.size()
        first_moment += (torch.mean(inputs, dim=(0, 2, 3)) - first_moment) / count
        sec_moment += (torch.mean(inputs ** 2, dim=(0, 2, 3)) - sec_moment) / count
        count += n * h * w
    mean = first_moment
    std = torch.sqrt(sec_moment - first_moment ** 2)
    return mean, std


def denormalise(images, mean, std, batched=True):
    num_channels = images.shape[1] if batched else images.shape[0]
    denom_images = torch.empty_like(images)
    for c in range(num_channels):
        denom_images[:, c, :, :] = images[:, c, :, :] * std[c] + mean[c]
    return denom_images


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def save_checkpoint(state, is_best, model_dir):
    filename = os.path.join(model_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, "model_best.pth.tar"))
