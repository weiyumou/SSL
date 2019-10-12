import torch
import torch.nn as nn
import torchvision


class Flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class TwoHeadedFC(nn.Module):

    def __init__(self, fc_in, fc1_out, fc2_out):
        super(TwoHeadedFC, self).__init__()
        self.fc1 = nn.Linear(in_features=fc_in, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc_in, out_features=fc2_out)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(x)
        return fc1_out, fc2_out


class ResNet18(nn.Module):

    def __init__(self, num_patches, num_angles):
        super(ResNet18, self).__init__()
        self.num_conv_features = 512
        self.backend = torchvision.models.resnet18()
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.num_conv_features, out_features=1024, bias=False),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches)
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
            nn.Linear(in_features=self.num_conv_features, out_features=num_classes)
        )
        self._initialise_fc()

    def _initialise_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
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
            nn.Linear(in_features=2048, out_features=1024, bias=False),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self._initialise_fc()


class ResNext50(ResNet18):

    def __init__(self, num_patches, num_angles):
        super(ResNext50, self).__init__(num_patches, num_angles)

        self.backend = torchvision.models.resnext50_32x4d()
        self.backend.fc = nn.Identity()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=2048, out_features=1024, bias=False),
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
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(inplace=True)
        # )
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=512 * 12 * 12, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
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


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class DRN_A(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x):
        for _, layer in self.named_children():
            x = layer(x)
        return x


class DRN_A_18(nn.Module):
    def __init__(self, num_patches, num_angles):
        super().__init__()
        self.backend = DRN_A(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self._initialise_fc()

    def forward(self, x):
        x = self.backend(x)
        return self.fc(x)

    def _initialise_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VGG11(nn.Module):
    def __init__(self, num_patches, num_angles):
        super(VGG11, self).__init__()

        self.backend = torchvision.models.vgg11_bn()
        self.backend.classifier = nn.Sequential()
        self.backend.avgpool = nn.Sequential()
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_patches * num_angles)
        )
        self._initialise_fc()

    def forward(self, x):
        for _, layer in self.backend.named_children():
            x = layer(x)
        return self.fc(x)

    def _initialise_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
