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
import sl_train
import train
from data import SSLTrainDataset, SSLValDataset

# mean and std for STL-10 Unlabeled train split
mean = (0.4226, 0.4120, 0.3636)
std = (0.2615, 0.2545, 0.2571)


def parse_args():
    parser = argparse.ArgumentParser(description='STL-10 SSL')
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
    parser.add_argument('--do_ssl',
                        help='Whether to do SSL',
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
    parser.add_argument('--do_sl',
                        help='Whether to do SL',
                        action="store_true")
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs for SL',
                        default=30)
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='Train batch size for SL',
                        default=128)
    parser.add_argument('--val_batch_size',
                        type=int,
                        help='Val batch size for SL',
                        default=128)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='Test batch size for SL',
                        default=128)
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
    input_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])

    model = models.ResNet18(args.num_patches, args.num_angles)
    model = torch.nn.DataParallel(model)

    if args.do_ssl:
        stl_unlabeled = datasets.STL10(root=args.data_dir,
                                       split='unlabeled',
                                       transform=input_transforms,
                                       download=args.download)
        indices = list(range(len(stl_unlabeled)))
        train_indices = indices[:int(len(indices) * 0.9)]
        val_indices = indices[int(len(indices) * 0.9):]
        dataloaders = {
            "train": DataLoader(
                SSLTrainDataset(Subset(stl_unlabeled, train_indices), args.num_patches, args.num_angles,
                                args.poisson_rate),
                shuffle=True, batch_size=args.ssl_train_batch_size, pin_memory=True),
            "val": DataLoader(
                SSLValDataset(Subset(stl_unlabeled, val_indices), args.num_patches, args.num_angles),
                shuffle=False, batch_size=args.ssl_val_batch_size, pin_memory=True)
        }

        # checkpoint = torch.load(os.path.join(args.model_dir, f"{args.model_name}"),
        #                         map_location=lambda storage, loc: storage.cuda(0))
        # model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{args.model_name}")))
        # dataloaders["train"].dataset.set_poisson_rate(args.poisson_rate)
        args.mean, args.std = mean, std
        # train.gen_grad_map(device, model, dataloaders["val"], args)

        model, best_val_accuracy = train.ssl_train(device, model, dataloaders, args)
        model_name = time.ctime().replace(" ", "_").replace(":", "_")
        model_name = f"{model_name}_{best_val_accuracy:.4f}.pt"
        torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))

    if args.do_sl:
        if args.model_name is None:
            raise ValueError("Model name must be specified")

        stl_train = datasets.STL10(root=args.data_dir,
                                   split='train',
                                   transform=input_transforms,
                                   download=args.download)

        args.num_classes = len(stl_train.classes)
        fold_indices = sl_train.stl_get_train_folds(os.path.join(args.data_dir, "stl10_binary/fold_indices.txt"))

        stl_test = datasets.STL10(root=args.data_dir,
                                  split='test',
                                  transform=input_transforms,
                                  download=args.download)
        dataloaders = {"test": DataLoader(stl_test, shuffle=False, batch_size=args.test_batch_size, pin_memory=True)}

        checkpoint = torch.load(os.path.join(args.model_dir, f"{args.model_name}"),
                                map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{args.model_name}")))
        # model.init_classifier(args.num_classes, freeze_params=False)

        args.mean, args.std = mean, std
        query_img, _ = stl_train[-1]
        dataloader = DataLoader(stl_train, batch_size=128, shuffle=False, pin_memory=True)
        top_images, top_labels = train.retrieve_topk_images(device, model, query_img, dataloader, args)

        # avg_test_accuracy = sl_train.stl_sl_train(device, model, stl_train, fold_indices, dataloaders, args)
        # utils.logger.info(f"Average Test Accuracy = {avg_test_accuracy}")


if __name__ == '__main__':
    main()
