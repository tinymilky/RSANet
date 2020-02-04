import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter

 # type: GN, BN, IN, LN
 # Specifically, GN with GN GN_2 GN_4 ...
def getNormModuleND(dimension = 3, norm_type = 'GN'):

    norm_ND = None
    params = []

    if 'GN' in norm_type:
        norm_ND = nn.GroupNorm
        if norm_type == 'GN':
            params.append(2) # default number of groups in GN
        else:
            eles = norm_type.split('_')
            params.append(int(eles[1]))
    else:
        if dimension == 1:
            norm_ND = nn.BatchNorm1d
        elif dimension == 2:
            norm_ND = nn.BatchNorm2d
        elif dimension == 3:
            norm_ND = nn.BatchNorm3d
    
    return norm_ND, params

class convNormAct(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, kernel_size, stride, padding, is_act = True):

        super(convNormAct, self).__init__()

        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)

        self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding, bias = False)
        self.norm = norm_3d(*(norm_params + [out_cs]))
        self.relu = nn.ReLU(inplace = True)
        self.is_act = is_act

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        if self.is_act:
            x = self.relu(x)
        
        return x

class normActConv(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, kernel_size, stride, padding):

        super(normActConv, self).__init__()

        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)

        self.norm = norm_3d(*(norm_params + [in_cs]))
        self.relu = nn.ReLU(inplace = True)
        self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding, bias = False)

    def forward(self, x):

        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        
        return x

class unetResBlockV1(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, downsample):

        super(unetResBlockV1, self).__init__()

        stride = 2 if downsample else 1

        if in_cs != out_cs or stride != 1:
            norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False),
                norm_3d(*(norm_params + [out_cs])),
            )
        else:
            self.shortcut = nn.Sequential() 

        self.layer1 = convNormAct(in_cs, out_cs, norm_type, 3, stride, 1, is_act = True)
        self.layer2 = convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, is_act = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.layer2(self.layer1(x)) + self.shortcut(x)

class unetResBlockV2(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, downsample):

        super(unetResBlockV2, self).__init__()

        stride = 2 if downsample else 1

        if in_cs != out_cs or stride != 1:
            norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
            self.shortcut = nn.Sequential(
                norm_3d(*(norm_params + [in_cs])),
                nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False)
            )
        else:
            self.shortcut = nn.Sequential()  
            
        self.layer1 = normActConv(in_cs, out_cs, norm_type, 3, stride, 1)
        self.layer2 = normActConv(out_cs, out_cs, norm_type, 3, 1, 1)

    def forward(self, x):

        return self.layer2(self.layer1(x)) + self.shortcut(x)

class unetVggBlock(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, num_convs = 2): 

        super(unetVggBlock, self).__init__()

        convs = []
        convs.append(convNormAct(in_cs, out_cs, norm_type, 3, 1, 1))
        for _ in range(num_convs-1):
            convs.append(convNormAct(out_cs, out_cs, norm_type, 3, 1, 1))

        self.layer = nn.Sequential(*convs)

    def forward(self, x):

        return self.layer(x)

class unetConvBlock(nn.Module):
    
    '''
    Three types of convolutions are supported.
    vgg: with control of number of convolution (num_convs, default 2) in one vgg conv layer
    resV1: resnet with post activation
    resV2: resnet with pre activation
    '''

    def __init__(self, in_cs, out_cs, norm_type, num_convs = 2, conv_type = 'vgg', downsample = False): 

        super(unetConvBlock, self).__init__()

        if conv_type == 'vgg':
            self.layer = unetVggBlock(in_cs, out_cs, norm_type, num_convs = num_convs)
        elif conv_type == 'resV1':
            self.layer = unetResBlockV1(in_cs, out_cs, norm_type, downsample)
        elif conv_type == 'resV2':
            self.layer = unetResBlockV2(in_cs, out_cs, norm_type, downsample)
            
    def forward(self, x):

        return self.layer(x)

class unetDownSample(nn.Module):

    def __init__(self, channels, down_type = 'conv', norm_type = 'GN_8'):

        super(unetDownSample, self).__init__()

        if down_type == 'conv':
            self.down = nn.Conv3d(channels, channels, 3, 2, 1, bias = False)
        if down_type == 'resV1':
            self.down = unetResBlockV1(channels, channels, norm_type, downsample = True)
        if down_type == 'resV2':
            self.down = unetResBlockV2(channels, channels, norm_type, downsample = True)
        if down_type == 'maxpool':
            self.down = nn.MaxPool3d(kernel_size = 2, padding = 1)

    def forward(self, x):

        return self.down(x)

class unetUpConv(nn.Module):

    def __init__(self, in_cs, out_cs, is_deconv, upsample_type):

        super(unetUpConv, self).__init__()

        if is_deconv:
            self.up = nn.Sequential(
                nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ConvTranspose3d(in_cs, out_cs, kernel_size=2, stride=2, padding=0, bias=False)            
            )
        else:
            if upsample_type == 'nearest':
                self.up = nn.Sequential(
                    nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Upsample(scale_factor=2),   
                )         
            else:
                self.up = nn.Sequential(
                    nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Upsample(scale_factor=2, mode = upsample_type, align_corners = True),   
                )   
    
    def forward(self, x):

        return self.up(x)

class unetPadCat(nn.Module):

    def __init__(self):

        super(unetPadCat, self).__init__()

    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)

        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        return torch.cat([leftIn, rightIn], 1)

class unetUpPadCatConv(nn.Module):
    
    def __init__(self, left_cs, right_cs, is_deconv, norm_type, conv_type = 'vgg', upsample_type = 'nearest'):

        super(unetUpPadCatConv, self).__init__()

        self.up = unetUpConv(right_cs, right_cs // 2, is_deconv, upsample_type)
        self.padCat = unetPadCat()
        self.conv = unetConvBlock(right_cs, left_cs, norm_type, conv_type = conv_type)
    
    def forward(self, left_x, right_x):

        right_x = self.up(right_x)
        x = self.padCat(left_x, right_x)
        del left_x, right_x

        return self.conv(x)