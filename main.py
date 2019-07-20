import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
import utils
import os
import torch
from torch.utils.data import DataLoader
import models
import train
import evaluation
import torch.nn as nn


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
    parser.add_argument('--do_cv',
                        help='Whether to do CV train',
                        action="store_true")
    parser.add_argument('--cv_epochs',
                        type=int,
                        help='Number of epochs to train for cross validation',
                        default=30)
    parser.add_argument('--cv_train_batch_size',
                        type=int,
                        help='Train batch size for cross validation',
                        default=64)
    parser.add_argument('--cv_val_batch_size',
                        type=int,
                        help='Val batch size for cross validation',
                        default=64)
    parser.add_argument('--cv_test_batch_size',
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

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_transforms = transforms.Compose([transforms.ToTensor()])

    stl_unlabeled = dset.STL10(root=args.data_dir,
                               split='unlabeled',
                               transform=input_transforms,
                               download=True)
    _, img_h, img_w = stl_unlabeled[0][0].size()
    unlabeled_dataloder = DataLoader(stl_unlabeled, shuffle=True, batch_size=args.ssl_batch_size)
    modules = models.build_VGG(img_h, img_w)

    if args.do_ssl:
        for index, (block, downsampler) in enumerate(zip(modules["blocks"], modules["downsamplers"])):
            preprocess = nn.Sequential(*modules["blocks"][:index])
            modules["blocks"][index], modules["downsamplers"][index] = \
                train.ssl_train(device, block, downsampler, unlabeled_dataloder, args.ssl_epochs, preprocess)
            torch.save(modules["blocks"][index].state_dict(), os.path.join(args.model_dir, f"block{index}.pt"))
            torch.save(modules["downsamplers"][index].state_dict(), os.path.join(args.model_dir, f"downsampler{index}.pt"))
            break

    if args.do_cv:
        stl_train = dset.STL10(root=args.data_dir,
                               split='train',
                               transform=input_transforms,
                               download=True)
        stl_test = dset.STL10(root=args.data_dir,
                              split='test',
                              transform=input_transforms,
                              download=True)

        num_classes = len(stl_train.classes)
        fold_indices = utils.get_train_folds(os.path.join(args.data_dir, "stl10_binary/fold_indices.txt"))

        test_dataloader = DataLoader(stl_test, shuffle=False, batch_size=args.cv_test_batch_size)

        # for index, (block, downsampler) in enumerate(zip(modules["blocks"], modules["downsamplers"])):
        #     block.load_state_dict(torch.load(os.path.join(args.model_dir, f"block{index}.pt")))
        #     downsampler.load_state_dict(torch.load(os.path.join(args.model_dir, f"downsampler{index}.pt")))

        modules["classifier"] = models.VGGClassifier(args.hidden_size, num_classes)

        best_model, best_accuracy = train.cv_train(device, stl_train, fold_indices, modules, args.cv_epochs,
                                                   args.cv_train_batch_size, args.cv_val_batch_size, num_classes,
                                                   test_dataloader)

        eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, best_model, test_dataloader)
        utils.logger.info(f"Test Accuracy = {eval_accuracy}")
        utils.logger.info(f"Test Loss = {eval_loss}")


if __name__ == '__main__':
    main()
