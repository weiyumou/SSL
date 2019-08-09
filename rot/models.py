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
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=num_patches * num_angles)
        )

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def init_classifier(self, num_classes):
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.fc[-1].in_features, out_features=num_classes)
        )


class RotResNet50(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet50()
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=2048 * 3 * 3, out_features=num_patches * num_angles)
        )

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def init_classifier(self, num_classes):
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.fc[-1].in_features, out_features=num_classes)
        )


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
        self.backend.fc = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * num_patches, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )

    def forward(self, x):
        n, c, h, w, _ = x.size()
        x = x.reshape(-1, c, h, w)
        for _, layer in self.backend.named_children():
            x = layer(x)
        x = x.reshape(n, -1)
        return self.fc(x)

    def init_classifier(self, num_classes):
        self.fc = nn.Linear(in_features=512, out_features=num_classes)


class RotAlexnetBN(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc = nn.Sequential(
            Flatten(),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(),
            nn.Linear(in_features=256 * 6 * 6, out_features=num_patches * num_angles)
        )

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x

    def init_classifier(self, num_classes):
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=num_classes)
        )


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
