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
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.backend.fc.in_features, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self.backend.fc = nn.Sequential()
        self._initialize_weights()

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def get_feature_maps(self, x):
        fmaps = {}
        for name, layer in self.backend.named_children():
            if "layer" in name:
                fmaps[f"{name}_input"] = x
            x = layer(x)
        return fmaps

    def init_classifier(self, num_classes, freeze_params=True):
        if freeze_params:
            for param in self.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.fc[1].in_features, out_features=num_classes)
        )
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class RotResNet50(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet50()
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_patches * num_angles)
        )
        self._initialize_weights()

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def init_classifier(self, num_classes):
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class RotAlexnetBN(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
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
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        # self._initialize_weights()

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x

    def init_classifier(self, num_classes):
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=num_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SimpleConv(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=64 * 3 * 3, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_patches * num_angles)
        )
        self._initialize_weights()

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
