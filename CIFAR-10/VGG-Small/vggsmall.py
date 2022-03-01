import torch
import torch.nn as nn
from binary_modules import BinarizeConv2d
# from util.ReAct_Modules import HardBinaryConv
import math
import torch.nn.init as init

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

class VGG_SMALL_1W1A(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = OwnBinaryConv(128, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = OwnBinaryConv(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = OwnBinaryConv(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = OwnBinaryConv(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = OwnBinaryConv(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model