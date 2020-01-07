from collections import namedtuple

import torch
from torchvision import models


class ResNet18(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(ResNet18, self).__init__()
        self.resnet18_pretrained = models.resnet18(pretrained=True)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        x = self.resnet18_pretrained.conv1(X)
        x = self.resnet18_pretrained.bn1(x)
        x = self.resnet18_pretrained.relu(x)
        x = self.resnet18_pretrained.maxpool(x)

        h_layer1 = self.resnet18_pretrained.layer1(x)
        h_layer2 = self.resnet18_pretrained.layer2(h_layer1)
        h_layer3 = self.resnet18_pretrained.layer3(h_layer2)
        h_layer4 = self.resnet18_pretrained.layer4(h_layer3)

        resnet_outputs = namedtuple("ResNetOutputs", ['layer1', 'layer2', 'layer3', 'layer4'])
        out = resnet_outputs(h_layer1, h_layer2, h_layer3, h_layer4)

        return out
