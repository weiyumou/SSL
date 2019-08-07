import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from fmap import models, train
import utils


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
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
                        action="store_true")
    parser.add_argument('--do_ssl',
                        help='Whether to do SSL',
                        action="store_true")
    parser.add_argument('--ssl_batch_size',
                        type=int,
                        help='Batch size for SSL',
                        default=64)
    parser.add_argument('--ssl_epochs',
                        type=int,
                        help='Number of epochs for SSL',
                        default=20)
    parser.add_argument('--do_sl',
                        help='Whether to do CV train',
                        action="store_true")
    parser.add_argument('--sl_epochs',
                        type=int,
                        help='Number of epochs to train for cross validation',
                        default=30)
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='Train batch size for cross validation',
                        default=64)
    parser.add_argument('--val_batch_size',
                        type=int,
                        help='Val batch size for cross validation',
                        default=64)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='Test batch size for cross validation',
                        default=64)
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Number of hidden units for classifier',
                        default=2048)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transforms = transforms.Compose([transforms.ToTensor()])

    modules = None
    if args.do_ssl:
        stl_unlabeled = dset.STL10(root=args.data_dir,
                                   split='unlabeled',
                                   transform=input_transforms,
                                   download=True)
        unlabeled_dataloader = DataLoader(stl_unlabeled, shuffle=False, batch_size=args.ssl_batch_size)

        _, img_h, img_w = stl_unlabeled[0][0].size()
        modules = models.build_VGG(img_h, img_w)

        for index, (block, downsampler) in enumerate(zip(modules["blocks"], modules["downsamplers"])):
            preprocess = nn.Sequential(*modules["blocks"][:index])
            modules["blocks"][index], modules["downsamplers"][index] = \
                train.ssl_train(device, block, downsampler, unlabeled_dataloader, args.ssl_epochs, preprocess)
            torch.save(modules["blocks"][index].state_dict(), os.path.join(args.model_dir, f"block{index}.pt"))
            torch.save(modules["downsamplers"][index].state_dict(),
                       os.path.join(args.model_dir, f"downsampler{index}.pt"))
            break

    if args.do_sl:
        stl_train = dset.STL10(root=args.data_dir,
                               split='train',
                               transform=input_transforms,
                               download=True)

        num_classes = len(stl_train.classes)
        fold_indices = utils.stl_get_train_folds(os.path.join(args.data_dir, "stl10_binary/fold_indices.txt"))

        stl_test = dset.STL10(root=args.data_dir,
                              split='test',
                              transform=input_transforms,
                              download=True)
        test_dataloader = DataLoader(stl_test, shuffle=False, batch_size=args.test_batch_size)

        # for index, (block, downsampler) in enumerate(zip(modules["blocks"], modules["downsamplers"])):
        #     block.load_state_dict(torch.load(os.path.join(args.model_dir, f"block{index}.pt")))
        #     downsampler.load_state_dict(torch.load(os.path.join(args.model_dir, f"downsampler{index}.pt")))

        if modules is None:
            _, img_h, img_w = stl_train[0][0].size()
            modules = models.build_VGG(img_h, img_w)
        modules["classifier"] = models.VGGClassifier(args.hidden_size, num_classes)

        avg_test_accuracy = train.sl_train(device, stl_train, fold_indices, modules, args.sl_epochs,
                                           args.train_batch_size, args.val_batch_size, num_classes,
                                           test_dataloader)

        utils.logger.info(f"Avg Test Accuracy = {avg_test_accuracy}")


if __name__ == '__main__':
    main()
