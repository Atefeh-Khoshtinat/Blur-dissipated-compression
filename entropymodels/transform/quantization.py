
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from entropymodels.layers.conv import conv1x1, conv3x3, conv, deconv
from entropymodels.layers.res_blk import *


class LatentResidualPredictionOld(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU):
        super().__init__()
        diff = abs(out_dim - in_dim)
        # such setting leads to much more parameters, you'd better use the setting in Minnen'20 ICIP paper.
        # To be fixed
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, in_dim - diff // 4),
            act(),
            conv3x3(in_dim - diff // 4, in_dim - diff // 2),
            act(),
            conv3x3(in_dim - diff // 2, in_dim - diff * 3 // 4),
            act(),
            conv3x3(in_dim - diff * 3 // 4, out_dim),
        )

    def forward(self, x):
        x = self.lrp_transform(x)
        x = 0.5 * torch.tanh(x)
        return x

class LatentResidualPrediction(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU):
        super().__init__()
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, 224),
            act(),
            conv3x3(224, 128),
            act(),
            conv3x3(128, out_dim),
        )

    def forward(self, x):
        x = self.lrp_transform(x)
        x = 0.5 * torch.tanh(x)
        return x



class CorrectLatentResidualPrediction(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU):
        super().__init__()
        # leads to 50M parameters reduction
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, 224),
            act(),
            conv3x3(224, 128),
            act(),
            conv3x3(128, out_dim),
        )

    def forward(self, x):
        x = self.lrp_transform(x)
        x = 0.5 * torch.tanh(x)
        return x
