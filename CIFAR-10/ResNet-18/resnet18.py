'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from binary_modules import BinarizeConv2d

class channel_w(nn.Module):
    def __init__(self,p):
        super(channel_w, self).__init__()
        self.w1 = torch.nn.Parameter(torch.rand(1)*0.001, requires_grad=True)

    def forward(self,x):
        output = self.w1 * x
        return output
class OwnBinaryConv(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super(OwnBinaryConv, self).__init__()
        self.shift1 = nn.Parameter(torch.zeros(1,in_ch,1,1), requires_grad=True)
        self.shift2 = nn.Parameter(torch.zeros(1, in_ch, 1, 1), requires_grad=True)
        self.conv = BinarizeConv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding)
        self.scale = channel_w(out_ch)

    def forward(self,x):
        x1 = x + self.shift1.expand_as(x)
        x2 = x + self.shift2.expand_as(x)

        out1 = self.conv(x1)
        out2 = self.conv(x2)

        out = out1+self.scale(out2)

        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = OwnBinaryConv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = OwnBinaryConv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],[64,128,256,512],**kwargs)

