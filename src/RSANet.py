import torch.nn as nn

from backbones.unet import unetConvBlock, unetDownSample, unetUpPadCatConv
from attModules.rsaModules import rsaBlock

class RSANet(nn.Module):
    def __init__(self,
        n_classes = 2,
        in_channels = 3,
        norm_type = 'GN_8', # default group norm with 8 groups
    ):
        super(RSANet, self).__init__()
        
        filters = [32, 64, 128, 256, 512]

        self.init_conv = nn.Conv3d(in_channels, filters[0],3,1,1, bias=False)
        self.dropout = nn.Dropout3d(0.2)

        self.conv1 = unetConvBlock(filters[0], filters[0], norm_type, num_convs=2)
        self.down1 = unetDownSample(filters[0], down_type = 'conv', norm_type=norm_type)

        self.conv2 = unetConvBlock(filters[0], filters[1], norm_type, num_convs=2)
        self.down2 = unetDownSample(filters[1], down_type = 'conv', norm_type=norm_type)
        
        self.conv3 = unetConvBlock(filters[1], filters[2], norm_type, num_convs=2)
        self.down3 = unetDownSample(filters[2], down_type = 'conv', norm_type=norm_type)

        self.conv4 = unetConvBlock(filters[2], filters[3], norm_type, num_convs=2)
        self.down4 = unetDownSample(filters[3], down_type = 'conv', norm_type=norm_type)
        
        self.center = unetConvBlock(filters[3], filters[4], norm_type, num_convs = 2)

        self.up_concat4 = unetUpPadCatConv(filters[3], filters[4], False, norm_type)
        self.up_concat3 = unetUpPadCatConv(filters[2], filters[3], False, norm_type)
        self.up_concat2 = unetUpPadCatConv(filters[1], filters[2], False, norm_type)
        self.up_concat1 = unetUpPadCatConv(filters[0], filters[1], False, norm_type)

        self.final = nn.Sequential(
            nn.Conv3d(filters[0], filters[0],3,1,1,bias=False),
            nn.Conv3d(filters[0], n_classes,1,bias=False),
        )    

        # --------- Recurrent slice-wise attention (RSA) module --------- #
        self.rsa_block = rsaBlock(filters[4])
        # --------- Recurrent slice-wise attention (RSA) module --------- #

    def forward(self, x):

        x = self.init_conv(x)
        x = self.dropout(x)

        conv1 = self.conv1(x)
        x = self.down1(conv1)

        conv2 = self.conv2(x)
        x = self.down2(conv2)
        
        conv3 = self.conv3(x)
        x = self.down3(conv3)
        
        conv4 = self.conv4(x)
        x = self.down4(conv4)
        
        x = self.center(x)

        # --------- RSA forward --------- #
        x = self.rsa_block(x)
        # --------- RSA forward--------- #

        x = self.up_concat4(conv4, x)
        del conv4
        x = self.up_concat3(conv3, x)
        del conv3
        x = self.up_concat2(conv2, x)
        del conv2
        x = self.up_concat1(conv1, x)
        del conv1

        x = self.final(x)

        return x