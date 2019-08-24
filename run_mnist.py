import argparse
import collections
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

import models
import train
from data import SSLTrainDataset, SSLValDataset


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST-Large SSL')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the data folder',
                        required=True)
    parser.add_argument('--model_dir',
                        type=str,
                        help='Path to the saved models',
                        default="models")
    parser.add_argument('--model_name',
                        type=str,
                        help='Name of the saved model',
                        default=None)
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
                        action="store_true")
    parser.add_argument('--ssl_train_batch_size',
                        type=int,
                        help='Train batch size for SSL',
                        default=64)
    parser.add_argument('--ssl_val_batch_size',
                        type=int,
                        help='Val batch size for SSL',
                        default=64)
    parser.add_argument('--ssl_num_epochs',
                        type=int,
                        help='Number of epochs for SSL',
                        default=20)
    parser.add_argument('--num_angles',
                        type=int,
                        help='Number of rotation angles',
                        default=4)
    parser.add_argument('--num_patches',
                        type=int,
                        help='Number of patches to extract from an image',
                        default=9)
    parser.add_argument('--learn_prd',
                        type=int,
                        help='Number of epochs before providing harder examples',
                        default=10)
    parser.add_argument('--poisson_rate',
                        type=int,
                        help='The initial poisson rate lambda',
                        default=2)
    args = parser.parse_args()
    return args


class MNISTLarge(Dataset):
    mean, std = (0.1334,), (0.2921,)
    input_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    def __init__(self, data_path, num_patches) -> None:
        super(MNISTLarge, self).__init__()
        self.mnist = torchvision.datasets.MNIST(data_path, train=True,
                                                transform=MNISTLarge.input_transforms, download=True)
        self.num_patches = num_patches
        self.image_dict = collections.defaultdict(list)
        for index, (image, label) in enumerate(self.mnist):
            if label < 9:
                self.image_dict[label].append(index)

    def __getitem__(self, index: int):
        random.seed(index)
        images = []
        for label in sorted(self.image_dict.keys()):
            random_idx = random.choice(self.image_dict[label])
            images.append(self._get_image(random_idx))
        random.seed()
        _, img_h, img_w = images[0].size()
        image = torch.stack(images, dim=3).reshape(1, -1, self.num_patches)
        image = F.fold(image, output_size=img_h * int(math.sqrt(self.num_patches)),
                       kernel_size=img_h, stride=img_h).squeeze(dim=0)
        return image,

    def __len__(self) -> int:
        return 100000

    def _get_image(self, index):
        return self.mnist[index][0]


def main():
    args = parse_args()
    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.ResNet18(args.num_patches, args.num_angles)

    mnist_large = MNISTLarge(args.data_dir, args.num_patches)
    indices = list(range(len(mnist_large)))
    train_indices = indices[:int(len(indices) * 0.9)]
    val_indices = indices[int(len(indices) * 0.9):]
    dataloaders = {
        "train": DataLoader(
            SSLTrainDataset(Subset(mnist_large, train_indices), args.num_patches, args.num_angles),
            shuffle=True, batch_size=args.ssl_train_batch_size, pin_memory=True),
        "val": DataLoader(
            SSLValDataset(Subset(mnist_large, val_indices), args.num_patches, args.num_angles),
            shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True)
    }

    model, best_val_accuracy = train.ssl_train(device, model, dataloaders, args.ssl_num_epochs,
                                               args.num_patches, args.num_angles, MNISTLarge.mean, MNISTLarge.std,
                                               args.learn_prd, args.poisson_rate)
    model_name = time.ctime().replace(" ", "_").replace(":", "_")
    model_name = f"{model_name}_{best_val_accuracy:.4f}.pt"
    torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))


if __name__ == '__main__':
    main()
