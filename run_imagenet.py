import argparse
import os
import random
import time

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import models
import train
from data import SSLTrainDataset, SSLValDataset

# mean and std for STL-10 Unlabeled train split
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description='Imagenet SSL')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the data folder',
                        required=True)
    parser.add_argument('--model_dir',
                        type=str,
                        help='Path to the save/load models',
                        default="models")
    parser.add_argument('--model_name',
                        type=str,
                        help='Path to the save/load models',
                        default=None)
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
                        action="store_true")
    parser.add_argument('--ssl_train_batch_size',
                        type=int,
                        help='Batch size for SSL',
                        default=128)
    parser.add_argument('--ssl_val_batch_size',
                        type=int,
                        help='Batch size for SSL',
                        default=128)
    parser.add_argument('--ssl_num_epochs',
                        type=int,
                        help='Number of epochs for SSL',
                        default=20)
    parser.add_argument('--num_angles',
                        type=int,
                        help='Number of hidden units for classifier',
                        default=4)
    parser.add_argument('--num_patches',
                        type=int,
                        help='Number of hidden units for classifier',
                        default=9)
    parser.add_argument('--learn_prd',
                        type=int,
                        help='Number of epochs before providing harder examples',
                        default=10)
    parser.add_argument('--poisson_rate',
                        type=int,
                        help='The initial poisson rate parameter lambda',
                        default=6)
    parser.add_argument('--download',
                        help='Whether to download datasets',
                        action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transforms = transforms.Compose([
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # model = models.ResNet18(args.num_patches, args.num_angles)
    # model = models.AlexnetBN(args.num_patches, args.num_angles)
    model = models.ResNet50(args.num_patches, args.num_angles)
    # model = rot.models.SimpleConv(args.num_patches, args.num_angles)
    model_name = args.model_name

    imagenet_train = datasets.ImageNet(root=args.data_dir,
                                       split='train',
                                       transform=input_transforms,
                                       download=args.download)
    imagenet_val = datasets.ImageNet(root=args.data_dir,
                                     split='val',
                                     transform=input_transforms,
                                     download=args.download)
    indices = list(range(len(imagenet_train)))
    train_indices = indices[:int(len(indices) * 0.9)]
    val_indices = indices[int(len(indices) * 0.9):]
    dataloaders = {
        "train": DataLoader(
            SSLTrainDataset(Subset(imagenet_train, train_indices), args.num_patches, args.num_angles),
            shuffle=True, batch_size=args.ssl_train_batch_size, pin_memory=True),
        "val": DataLoader(
            SSLValDataset(Subset(imagenet_train, val_indices), args.num_patches, args.num_angles),
            shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True),
        "test": DataLoader(SSLValDataset(imagenet_val, args.num_patches, args.num_angles),
                           shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True)
    }

    model, best_val_accuracy = train.ssl_train(device, model, dataloaders, args.ssl_num_epochs,
                                               args.num_patches, args.num_angles, mean, std, args.learn_prd,
                                               args.poisson_rate)
    model_name = time.ctime().replace(" ", "_").replace(":", "_")
    model_name = f"{model_name}_{best_val_accuracy:.4f}.pt"
    torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))

    # model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{model_name}")))
    # query_img, _ = stl_train[-3]
    # dataloader = DataLoader(stl_train, batch_size=128, shuffle=False, pin_memory=True)
    # top_images, top_labels = train.retrieve_topk_images(device, model, query_img, dataloader, mean, std)


if __name__ == '__main__':
    main()
