"""Dilated ResNet
   source:  https://github.com/wuhuikai/FastFCN/blob/master/encoding/dilated/resnet.py
"""


import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo




__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, stride=stride,
                     padding=1, bias=False)



class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=previous_dilation, dilation=previous_dilation,bias=False)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)


        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck"""




    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride


    def _sum_each(self, x,y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z




    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    """Dilated Pre-Trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    block: Block
        Class for the residual block. Options are basicBlockV1, BottleneckV1
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."

    """

    def __init__(self, block, layer, num_class=1000, dilated=True,
                 deep_base=True, norm_layer=nn.BatchNorm2d, output_size=8):
       self.inplanes = 128 if deep_base else 64

       super(ResNet, self).__init__()

       if deep_base:
           self.conv1 = nn.Sequential(
               nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=1),
               norm_layer(64),
               nn.ReLU(inplace=True),
               nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
               norm_layer(64),
               nn.ReLU(inplace=True),
               nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
           )
       else:
           self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

       self.bn1 = norm_layer(self.inplanes)
       self.relu = nn.ReLU(inplace=True)
       self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

       self.layer1 = self._make_layer(block, 64, layer[0], norm_layer=norm_layer)
       self.layer2 = self._make_layer(block, 128, layer[1], stride=2, norm_layer=norm_layer)

       dilation_rate = 2

       if dilated and output_size <= 8:
           self.layer3 = self._make_layer(block, 256, layer[2], stride=1,
                                          dilation = dilation_rate, norm_layer= norm_layer)
           dilation_rate *=2
       else:
           self.layer3 = self._make_layer(block, 256, layer[2], stride=2, norm_layer = norm_layer)
        """To be  continue from line number:167
            source: https://github.com/wuhuikai/FastFCN/blob/master/encoding/dilated/resnet.py
            
        """









