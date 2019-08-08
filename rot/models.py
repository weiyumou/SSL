import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


class Flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class RotResNet18(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet18()
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.fc = nn.Sequential()
        self.fc = nn.Linear(in_features=512 * 3 * 3, out_features=num_patches * num_angles)

    def forward(self, x):
        backend_out = self.backend(x)
        return self.fc(torch.flatten(backend_out, start_dim=1))

    def init_classifier(self, num_classes):
        self.fc = nn.Linear(in_features=self.fc.in_features, out_features=num_classes)


class RotResNet50(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet50()
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.add_module("flatten", Flatten())
        self.backend.fc = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048 * 3 * 3, out_features=num_patches * num_angles)
        )

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def init_classifier(self, num_classes):
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)


class RotAlexnet(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()
        self.backend = torchvision.models.alexnet()
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.classifier = nn.Sequential()
        self.fc = nn.Linear(in_features=256 * 3 * 3, out_features=num_patches * num_angles)

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def init_classifier(self, num_classes):
        self.fc = nn.Linear(in_features=self.fc.in_features, out_features=num_classes)


class RotSiamResnet18(nn.Module):
    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet18()
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.fc = nn.Sequential()
        self.fc = nn.Linear(in_features=512 * 3 * 3, out_features=num_patches * num_angles)

    def forward(self, x):
        backend_out = self.backend(x)
        return self.fc(torch.flatten(backend_out, start_dim=1))

    def init_classifier(self, num_classes):
        self.fc = nn.Linear(in_features=self.fc.in_features, out_features=num_classes)


class SSLTrainDataset(Dataset):

    def __init__(self, train_dataset, num_patches, num_angles) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.num_patches = num_patches
        self.num_angles = num_angles

    def __getitem__(self, index: int):
        rotation = torch.empty(self.num_patches, dtype=torch.long).random_(self.num_angles)
        perm = torch.randperm(self.num_patches, dtype=torch.long)
        return self.train_dataset[index][0], rotation, perm

    def __len__(self) -> int:
        return len(self.train_dataset)


class SSLValDataset(Dataset):

    def __init__(self, val_dataset, num_patches, num_angles) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.rotations = dict()
        self.perms = dict()
        for index in range(len(val_dataset)):
            self.rotations[index] = torch.empty(num_patches, dtype=torch.long).random_(num_angles)
            self.perms[index] = torch.randperm(num_patches, dtype=torch.long)

    def __getitem__(self, index: int):
        return self.val_dataset[index][0], self.rotations[index], self.perms[index]

    def __len__(self) -> int:
        return len(self.val_dataset)
