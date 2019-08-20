import torch
import torch.nn as nn
import torchvision


class Flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class ResNet18(nn.Module):

    def __init__(self, num_patches, num_angles):
        super(ResNet18, self).__init__()

        self.backend = torchvision.models.resnet18()
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self._initialise_fc()

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
        self._initialise_fc()

    def _initialise_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNet50(ResNet18):

    def __init__(self, num_patches, num_angles):
        super(ResNet50, self).__init__(num_patches, num_angles)

        self.backend = torchvision.models.resnet50()
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self._initialise_fc()


class AlexnetBN(nn.Module):

    def __init__(self, num_patches, num_angles):
        super(AlexnetBN, self).__init__()
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
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_patches * num_angles)
        )
        # self._initialize_weights()

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x

    def init_classifier(self, num_classes, freeze_params=True):
        if freeze_params:
            for param in self.parameters():
                param.requires_grad = False

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