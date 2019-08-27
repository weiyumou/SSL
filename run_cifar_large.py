import argparse
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
    parser = argparse.ArgumentParser(description='CIFAR-Large SSL')
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


class CIFARLarge(Dataset):
    def __init__(self, dataset, num_patches, train) -> None:
        super(CIFARLarge, self).__init__()
        self.data_loader = DataLoader(dataset, shuffle=train, batch_size=num_patches, drop_last=True)
        self.data_iter = iter(self.data_loader)
        self.num_patches = num_patches

    def __getitem__(self, index: int):
        try:
            images, _ = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            images, _ = next(self.data_iter)

        *_, h, w = images.size()  # (9, 3, 32, 32)
        images = images.permute(1, 2, 3, 0)
        big_image = F.fold(images.reshape(1, -1, self.num_patches),
                           output_size=h * int(math.sqrt(self.num_patches)),
                           kernel_size=h, stride=h).squeeze(dim=0)
        return big_image,

    def __len__(self) -> int:
        return len(self.data_loader)


mean = (0.4883, 0.4739, 0.4334)
std = (0.2457, 0.2409, 0.2514)


def main():
    global mean, std
    args = parse_args()
    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    cifar = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=input_transforms, download=True)
    indices = list(range(len(cifar)))
    train_indices = indices[:int(len(indices) * 0.9)]
    val_indices = indices[int(len(indices) * 0.9):]

    train_set = CIFARLarge(Subset(cifar, train_indices), args.num_patches, train=True)
    val_set = CIFARLarge(Subset(cifar, val_indices), args.num_patches, train=False)
    dataloaders = {
        "train": DataLoader(SSLTrainDataset(train_set, args.num_patches, args.num_angles),
                            shuffle=False, batch_size=args.ssl_train_batch_size, pin_memory=True),
        "val": DataLoader(SSLValDataset(val_set, args.num_patches, args.num_angles),
                          shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True)
    }

    model = models.ResNet18(args.num_patches, args.num_angles)
    # model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{args.model_name}")))
    # train.gen_grad_map(device, model, dataloaders, args.num_patches, args.num_angles)

    model, best_val_accuracy = train.ssl_train(device, model, dataloaders, args)
    model_name = time.ctime().replace(" ", "_").replace(":", "_")
    model_name = f"{model_name}_{best_val_accuracy:.4f}.pt"
    torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))


if __name__ == '__main__':
    main()
