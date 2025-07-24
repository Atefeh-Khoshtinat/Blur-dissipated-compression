
import torch
import torch.nn as nn
from .network_components import ResnetBlock, VBRCondition, Downsample, Upsample, GDN1
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .utils import quantize
import numpy as np



def get_scale_table(min=0.11, max=256, levels=64):
    return np.exp(np.linspace(np.log(min), np.log(max), levels)).astype(float).tolist()



class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.vbr = vbr
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.reversed_dims = list(reversed([out_channels, *map(lambda m: dim * m, dim_mults)]))
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.reversed_hyper_dims = list(reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)]))
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_in_out = list(zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:]))

        self.g_a = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 3, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 3, dim * 4, kernel_size=5, stride=2, padding=2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(dim * 3, dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(dim, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=1, padding=1),
        )

        self.entropy_bottleneck = EntropyBottleneck(dim * 4)
        self.gaussian_conditional = GaussianConditional(get_scale_table())

    def encode(self, input, cond=None):
        y = self.g_a(input)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        state4bpp = {
            "latent": y,
            "hyper_latent": z,
            "scales_hat": scales_hat,
            "means_hat": means_hat,
            "z_hat": z_hat,
            "y_hat": y_hat,
            "z_likelihoods": z_likelihoods,
            "y_likelihoods": y_likelihoods
        }
        return y_hat, z_hat, state4bpp

    def decode(self, y_hat, cond=None):
        output = []
        input = y_hat
        for layer in self.g_s:
            input = layer(input)
            if isinstance(layer, nn.ConvTranspose2d):

                output.append(input)


        return output[::-1]



    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        z_bpp = -state4bpp["z_likelihoods"].log2()
        y_bpp = -state4bpp["y_likelihoods"].log2()
        return (y_bpp.sum(dim=(1, 2, 3)) + z_bpp.sum(dim=(1, 2, 3))) / (H * W)

    def forward(self, input, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input, cond)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent, cond)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }

class BigCompressor(Compressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim_mults=(1, 3, 3, 4), **kwargs)
        self.build_network()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, ),
                VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                Downsample(dim_out)
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                VBRCondition(1, dim_out if not is_last else dim_in) if self.vbr else nn.Identity(),
                Upsample(dim_out if not is_last else dim_in, dim_out)
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, 3, 1, 1) if ind == 0 else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if not is_last else nn.Identity()
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, 3, 1, 1) if is_last else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if not is_last else nn.Identity()
            ]))


class SimpleCompressor(Compressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim_mults=(1, 2, 3, 3), **kwargs)
        self.build_network()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                GDN1(dim_out) if not is_last else nn.Identity()
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                GDN1(dim_out, True) if not is_last else nn.Identity()
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, 3, 1, 1) if ind == 0 else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if not is_last else nn.Identity()
            ]))

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, 3, 1, 1) if is_last else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if not is_last else nn.Identity()
            ]))
