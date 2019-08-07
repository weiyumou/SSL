import logging

import matplotlib.pyplot as plt
import numpy as np

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
    plt.show()


def show_mask(mask):
    plt.figure()
    np_mask = mask.detach().cpu().numpy()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np_mask, cmap='gray')
