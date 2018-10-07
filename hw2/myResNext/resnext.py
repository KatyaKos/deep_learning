"""
Based on: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Article: https://arxiv.org/pdf/1611.05431.pdf
"""

import math
from torch import nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNext', 'resnext50', 'resnext101', 'resnext152']

model_urls = {
    'resnext101': 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl',
    'resnext152': 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl'
}


class ResNextBlock(nn.Module):

    expansion = 4

    """
    Parameter cardinality is drawn from the article.
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, multiplier=4, cardinality=32):
        super(ResNextBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.model = nn.Sequential(
            nn.Conv2d(inplanes, planes * multiplier, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * multiplier),
            self.relu,
            nn.Conv2d(planes * multiplier, planes * multiplier,
                      kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality),
            nn.BatchNorm2d(planes * multiplier),
            self.relu,
            nn.Conv2d(planes * multiplier, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.model(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNext(nn.Module):

    def __init__(self, block, layers, multiplier, cardinality=32, num_classes=1000):
        self.inplanes = 64
        self.cardinality = cardinality
        self.multiplier = multiplier
        super(ResNext, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample,
                  multiplier=self.multiplier, cardinality=self.cardinality))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, multiplier=self.multiplier, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNext(ResNextBlock, [3, 4, 6, 3], 4, **kwargs)
    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, [3, 4, 23, 3], 8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101']))
    return model


def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNext(ResNextBlock, [3, 8, 36, 3], 8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext152']))
    return model
