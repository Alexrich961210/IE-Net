import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()

        w = self.weight

        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)

    def forward(self, input):
        a = input
        w = self.weight
        w0 = w - w.mean([1, 2, 3], keepdim=True)
        w1 = w0 / torch.sqrt(w0.var([1, 2, 3], keepdim=True) + 1e-5)
        if self.training:
            a0 = a / torch.sqrt(a.var([1, 2, 3], keepdim=True) + 1e-5)
        else:
            a0 = a

        #* binarize
        bw = OwnQuantize().apply(w1,self.k.to(w.device),self.t.to(w.device))

        ba = OwnQuantize_a().apply(a0,self.k.to(w.device),self.t.to(w.device))

        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output

#-----------------------IEE------------------------------------------
class OwnQuantize_a(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        # grad_input = k * (1.4*t - torch.abs(t**2 * input))
        grad_input = k * (3*torch.sqrt(t**2/3) - torch.abs(t ** 2 * input*3)/2)
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None
class OwnQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        # grad_input = k * (1.4*t - torch.abs(t**2 * input))
        grad_input = k * (3*torch.sqrt(t**2/3) - torch.abs(t ** 2 * input*3)/2)
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None

#-----------------------------Bi-Real------------------------------
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

#
class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input
