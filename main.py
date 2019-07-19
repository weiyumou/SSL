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


def parse_args():
    parser = argparse.ArgumentParser(description='SSL')
    parser.add_argument('--data_path',
                        type=str,
                        help='Path to the data folder',
                        required=True)
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

    input_transforms = transforms.Compose([transforms.ToTensor()])
    # stl_unlabeled = dset.STL10(root=args.data_path,
    #                            split='unlabeled',
    #                            transform=input_transforms,
    #                            download=True)
    stl_train = dset.STL10(root=args.data_path,
                           split='train',
                           transform=input_transforms,
                           download=True)
    stl_test = dset.STL10(root=args.data_path,
                          split='test',
                          transform=input_transforms,
                          download=True)

    _, img_h, img_w = stl_train[0][0].size()
    num_classes = len(stl_train.classes)
    fold_indices = utils.get_train_folds(os.path.join(args.data_path, "stl10_binary/fold_indices.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataloader = DataLoader(stl_test, shuffle=False, batch_size=args.cv_test_batch_size)

    modules = models.build_VGG(img_h, img_w)
    modules["classifier"] = models.VGGClassifier(args.hidden_size, num_classes)

    best_model, best_accuracy = train.cv_train(device, stl_train, fold_indices, modules, args.cv_epochs,
                                               args.cv_train_batch_size, args.cv_val_batch_size, num_classes)

    eval_accuracy, eval_loss = evaluation.evaluate_classifier(device, best_model, test_dataloader)
    utils.logger.info(f"Test Accuracy = {eval_accuracy}")
    utils.logger.info(f"Test Loss = {eval_loss}")


if __name__ == '__main__':
    main()
