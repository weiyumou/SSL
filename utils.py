import numpy as np
import matplotlib.pyplot as plt
import logging
import random
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def show(img):
    plt.figure()
    npimg = img.detach().cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def show_mask(mask):
    plt.figure()
    np_mask = mask.detach().cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np_mask, cmap='gray')


def get_train_folds(fold_file, num_folds=10):
    fold_indices = np.fromfile(fold_file, dtype=int, sep=" ")
    return fold_indices.reshape(num_folds, -1)


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
