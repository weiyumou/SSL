import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import rot.models
import rot.train
import sl_train
import utils
import torchvision

# mean and std for STL-10 Unlabeled train split
mean = (0.4226, 0.4120, 0.3636)
std = (0.2615, 0.2545, 0.2571)


def parse_args():
    parser = argparse.ArgumentParser(description='SSL')
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
    parser.add_argument('--do_ssl',
                        help='Whether to do SSL',
                        action="store_true")
    parser.add_argument('--ssl_train_batch_size',
                        type=int,
                        help='Batch size for SSL',
                        default=64)
    parser.add_argument('--ssl_val_batch_size',
                        type=int,
                        help='Batch size for SSL',
                        default=64)
    parser.add_argument('--ssl_num_epochs',
                        type=int,
                        help='Number of epochs for SSL',
                        default=20)
    parser.add_argument('--do_sl',
                        help='Whether to do SL',
                        action="store_true")
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs to train for cross validation',
                        default=30)
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='Train batch size for cross validation',
                        default=128)
    parser.add_argument('--val_batch_size',
                        type=int,
                        help='Val batch size for cross validation',
                        default=128)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='Test batch size for cross validation',
                        default=128)
    parser.add_argument('--num_angles',
                        type=int,
                        help='Number of hidden units for classifier',
                        default=4)
    parser.add_argument('--num_patches',
                        type=int,
                        help='Number of hidden units for classifier',
                        default=9)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])

    model = rot.models.RotResNet18(args.num_patches, args.num_angles)
    # model = rot.models.RotAlexnetBN(args.num_patches, args.num_angles)
    # model = rot.models.RotResNet50(args.num_patches, args.num_angles)
    # model = rot.models.RotSiamResnet18(args.num_patches, args.num_angles)
    model_name = args.model_name

    if args.do_ssl:
        stl_unlabeled = datasets.STL10(root=args.data_dir,
                                       split='unlabeled',
                                       transform=input_transforms,
                                       download=True)
        num_examples = len(stl_unlabeled)
        indices = list(range(num_examples))
        train_indices = indices[:int(num_examples * 0.9)]
        val_indices = indices[int(num_examples * 0.9):]
        unlabeled_dataloaders = {
            "train": DataLoader(
                rot.models.SSLTrainDataset(Subset(stl_unlabeled, train_indices), args.num_patches, args.num_angles),
                shuffle=True,
                batch_size=args.ssl_train_batch_size,
                pin_memory=True),
            "val": DataLoader(
                rot.models.SSLValDataset(Subset(stl_unlabeled, val_indices), args.num_patches, args.num_angles),
                shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True)
        }

        # model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{model_name}")))

        data_iter = unlabeled_dataloaders["val"].__iter__()
        inputs, rotations, perms = next(data_iter)
        utils.show(torchvision.utils.make_grid(inputs, nrow=4, normalize=True, scale_each=True), mean, std)

        inputs, rotations = rot.train.random_rotate(inputs, args.num_patches, rotations, perms)
        utils.show(torchvision.utils.make_grid(inputs, nrow=4, normalize=True, scale_each=True), mean, std)

        model, best_val_accuracy = rot.train.ssl_train(device, model, unlabeled_dataloaders, args.ssl_num_epochs,
                                                       args.num_patches, args.num_angles)
        model_name = time.ctime().replace(" ", "_").replace(":", "_")
        model_name = f"{model_name}_{best_val_accuracy:.4f}.pt"
        torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))

    if args.do_sl:
        if model_name is None:
            raise ValueError("Model name must be specified")

        stl_train = datasets.STL10(root=args.data_dir,
                                   split='train',
                                   transform=input_transforms,
                                   download=True)

        num_classes = len(stl_train.classes)
        fold_indices = sl_train.stl_get_train_folds(os.path.join(args.data_dir, "stl10_binary/fold_indices.txt"))

        stl_test = datasets.STL10(root=args.data_dir,
                                  split='test',
                                  transform=input_transforms,
                                  download=True)
        test_dataloader = DataLoader(stl_test, shuffle=False, batch_size=args.test_batch_size, pin_memory=True)

        model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{model_name}")))
        model.init_classifier(num_classes)

        avg_test_accuracy = sl_train.stl_sl_train(device, model, stl_train, fold_indices, args.num_epochs,
                                                  args.train_batch_size, args.val_batch_size, num_classes,
                                                  test_dataloader)
        utils.logger.info(f"Average Test Accuracy = {avg_test_accuracy}")

        # inputs, rotations = next(unlabeled_dataloaders["val"].__iter__())
        # inputs, labels = utils.random_rotate(inputs, model.num_patches, model.num_angles, rotations)
        # utils.show(torchvision.utils.make_grid(inputs, nrow=4, normalize=True, scale_each=True))
        # model = model.to(device)
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        # outputs = model(inputs)
        # pass


if __name__ == '__main__':
    main()
