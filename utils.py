import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

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
    for c in range(num_channels):
        npimg[c, :, :] = npimg[c, :, :] * std[c] + mean[c]

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
