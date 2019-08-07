import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


class RotResNet(nn.Module):

    def __init__(self, num_patches, num_angles):
        super().__init__()

        self.backend = torchvision.models.resnet18()
        self.fc = nn.Linear(in_features=self.backend.fc.in_features * 3 * 3,
                            out_features=num_patches * num_angles)
        self.backend.avgpool = nn.AdaptiveAvgPool2d(output_size=3)
        self.backend.fc = nn.Sequential()

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
