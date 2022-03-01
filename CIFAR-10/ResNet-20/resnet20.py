import torch
import torch.nn as nn
import torch.nn.functional as F
from binary_modules import BinarizeConv2d
import torch.nn.init as init


# from torchsummary import summary

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class channel_w(nn.Module):
    def __init__(self,out_ch):
        super(channel_w, self).__init__()
        self.w1 =torch.nn.Parameter(torch.rand(1,out_ch,1,1)*0.001,requires_grad=True)

    def forward(self,x):
        out = self.w1 * x
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.shift1 = nn.Parameter(torch.zeros(1,inplanes,1,1), requires_grad=True)
        self.shift2 = nn.Parameter(torch.zeros(1, inplanes, 1, 1), requires_grad=True)

        self.binary_conv1 = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.scale = channel_w(planes)

        self.hardtanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample

        self.inplanes = inplanes

        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = x + self.shift1.expand_as(x)
        x2 = x + self.shift2.expand_as(x)

        out1 = self.binary_conv1(x1)
        out2 = self.binary_conv1(x2)


        out = out1 + self.scale(out2)

        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.hardtanh(out)

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(BiRealNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        pad = 0 if planes == self.inplanes else planes // 4
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                        nn.AvgPool2d((2,2)),
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.hardtanh(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)

        return x


def resnet20_1w1a(num_classes=10):
    return BiRealNet(BasicBlock, [6, 6, 6], num_classes=num_classes)
