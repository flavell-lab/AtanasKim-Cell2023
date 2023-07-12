# adapted from https://github.com/milesial/Pytorch-UNet
import torch.nn.functional as F
from .unet_part import *

class UNet2D(nn.Module):
    def __init__(self, n_ch_in=1, n_class=1, n_feature=64, bilinear=True):
        super(UNet2D, self).__init__()
        self.n_ch_in = n_ch_in
        self.n_class = n_class
        self.bilinear = bilinear

        self.inc = DoubleConv(n_ch_in, n_feature)
        self.down1 = Down(n_feature, n_feature * 2)
        self.down2 = Down(n_feature * 2, n_feature * 4)
        self.down3 = Down(n_feature * 4, n_feature * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_feature * 8, n_feature * 16 // factor)
        self.up1 = Up(n_feature * 16, n_feature * 8, bilinear)
        self.up2 = Up(n_feature * 8, n_feature * 4, bilinear)
        self.up3 = Up(n_feature * 4, n_feature * 2, bilinear)
        self.up4 = Up(n_feature * 2, n_feature * factor, bilinear)
        self.outc = OutConv(n_feature, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
