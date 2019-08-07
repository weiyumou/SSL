import torch.nn as nn
import torch.optim as optim


class VGGBlock(nn.Module):
    def __init__(self, num_conv_layers, num_in_channels, num_out_channels):
        super(VGGBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=num_in_channels,
                            out_channels=num_out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(num_features=num_out_channels),
                  nn.ReLU()]
        for idx in range(num_conv_layers - 1):
            layers.extend([nn.Conv2d(in_channels=num_out_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3, stride=1, padding=1, bias=False),
                           nn.BatchNorm2d(num_features=num_out_channels),
                           nn.ReLU()])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        activations = [self.layers[0](x)]
        for index in range(1, len(self.layers)):
            activations.append(self.layers[index](activations[-1]))
        return activations

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Downsampler(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, fmap_h, fmap_w):
        super(Downsampler, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=num_out_channels,
                      out_channels=num_in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=num_in_channels),
            nn.ReLU()
        )

        self.fc = nn.Linear(in_features=fmap_h * fmap_w // 4,
                            out_features=fmap_h * fmap_w // 4)

        self._initialize_weights()

    # def forward(self, x):
    #     conv_out = self.conv1x1(x)
    #     n, c, h, w = conv_out.size()
    #     fc_out = self.fc(conv_out.reshape(n, c, -1))
    #     return fc_out.reshape(n, c, h, w)

    def forward(self, x):
        activations = [self.conv1x1[0](x)]
        for index in range(1, len(self.conv1x1)):
            activations.append(self.conv1x1[index](activations[-1]))
        conv_out = activations[-1]
        n, c, h, w = conv_out.size()
        fc_out = self.fc(conv_out.reshape(n, c, -1))
        activations.append(fc_out.reshape(n, c, h, w))
        return activations

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.8)
                nn.init.constant_(m.bias, -0.2)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)


class VGGClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(VGGClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * 3 * 3, out_features=hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_classes, bias=True))
        self._initialize_weights()

    def forward(self, x):
        pool_out = self.pool(x)
        fc1_out = self.fc1(pool_out.reshape(pool_out.size(0), -1))
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        return fc3_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def build_VGG(fmap_h, fmap_w):
    num_in_channels, num_out_channels = 3, 64
    num_conv_layers = [2, 2, 3, 3, 3]
    vgg_blocks = [VGGBlock(num_conv_layers[0], num_in_channels, num_out_channels)]
    downsamplers = [Downsampler(num_in_channels, num_out_channels, fmap_h, fmap_w)]

    num_in_channels, num_out_channels = 64, 128
    for idx in range(1, len(num_conv_layers)):
        fmap_h //= 2
        fmap_w //= 2
        vgg_blocks.append(VGGBlock(num_conv_layers[idx], num_in_channels, num_out_channels))
        downsamplers.append(Downsampler(num_in_channels, num_out_channels, fmap_h, fmap_w))
        num_in_channels, num_out_channels = num_out_channels, min(512, num_out_channels * 2)

    return {"blocks": vgg_blocks, "downsamplers": downsamplers}


def build_optimisers(device, modules):
    optimisers = {"blocks": [], "downsamplers": []}
    for idx in range(len(modules["blocks"])):
        modules["blocks"][idx].train().to(device)
        modules["downsamplers"][idx].train().to(device)
        optimisers["blocks"].append(optim.Adam(modules["blocks"][idx].parameters()))
        optimisers["downsamplers"].append(optim.Adam(modules["downsamplers"][idx].parameters()))

    modules["classifier"].train().to(device)
    optimisers["classifier"] = optim.Adam(modules["classifier"].parameters())
    return optimisers
