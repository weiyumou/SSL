import numpy as np
import matplotlib.pyplot as plt
import logging

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
