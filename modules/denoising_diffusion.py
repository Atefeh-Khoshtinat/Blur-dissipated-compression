import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lpips
import numpy as np

# --- DCT and IDCT Utilities --- #
def dct_2d(x):
    return torch.fft.fft2(x, norm='ortho')

def idct_2d(x):
    return torch.fft.ifft2(x, norm='ortho').real

# --- Physics-inspired Scaling --- #
def get_frequency_scaling(t, img_dim=256, sigma_blur_max=1.0, sigma_t=1.0, min_scale=0.001):
    dissipation_time = sigma_t ** 2 / 2
    freqs = math.pi * torch.linspace(0, img_dim - 1, img_dim, device=t.device) / img_dim
    lambda_ = freqs[None, :, None, None] ** 2 + freqs[None, None, :, None] ** 2
    scaling = torch.exp(-lambda_ * dissipation_time) * (1 - min_scale)
    return scaling + min_scale

def get_noise_scaling_cosine(t, logsnr_min=-10, logsnr_max=10):
    limit_max = torch.atan(torch.exp(torch.tensor(-0.5 * logsnr_max, device=t.device)))
    limit_min = torch.atan(torch.exp(torch.tensor(-0.5 * logsnr_min, device=t.device))) - limit_max
    logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max))
    return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr))

def get_alpha_sigma(t, img_dim, channels=3):
    B = t.shape[0]
    freq_scaling = get_frequency_scaling(t, img_dim=img_dim).squeeze(-1)
    freq_scaling = freq_scaling.expand(B, img_dim, img_dim).unsqueeze(1)
    freq_scaling = freq_scaling.expand(B, channels, img_dim, img_dim)
    a, sigma = get_noise_scaling_cosine(t)
    alpha = a.view(B, 1, 1, 1) * freq_scaling
    sigma = sigma.view(B, 1, 1, 1)
    return alpha.clamp(min=1e-6), sigma.clamp(min=1e-6)

# --- Diffusion Forward --- #
def diffuse(x, t):
    x_freq = dct_2d(x)
    alpha, sigma = get_alpha_sigma(t, img_dim=x.shape[-1], channels=x.shape[1])
    eps = torch.randn_like(x)
    z_t = idct_2d(alpha * x_freq.real) + sigma * eps
    return z_t, eps

# --- Denoising Reverse --- #
def denoise(z_t, t, T, model, delta=1e-8, img_dim=None):
    if z_t.dim() == 5:
        B1, B2, C, H, W = z_t.shape
        z_t = z_t.view(B1 * B2, C, H, W)
        t = t.view(B1 * B2, 1)

    B, C, H, W = z_t.shape
    img_dim = W if img_dim is None else img_dim
    alpha_s, sigma_s = get_alpha_sigma(t - 1 / T, img_dim, channels=C)
    alpha_t, sigma_t = get_alpha_sigma(t, img_dim, channels=C)

    alpha_ts = alpha_t / torch.clamp(alpha_s, min=delta)
    alpha_st = 1.0 / torch.clamp(alpha_ts, min=delta)
    sigma2_ts = sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2
    sigma2_ts = torch.clamp(sigma2_ts, min=delta)

    denom1 = torch.clamp(sigma_s ** 2, min=delta)
    denom2 = torch.clamp(sigma_t ** 2 / torch.clamp(alpha_ts ** 2, min=delta) - sigma_s ** 2, min=delta)
    sigma2_denoise = 1 / torch.clamp(1 / denom1 + 1 / denom2, min=delta)

    coeff1 = alpha_ts * sigma2_denoise / (sigma2_ts + delta)
    coeff2 = alpha_st * sigma2_denoise / denom1

    hat_eps = model(z_t, t)
    u_t = dct_2d(z_t).real
    u_eps = dct_2d(hat_eps).real

    mu_denoise = idct_2d(coeff1 * u_t) + idct_2d(coeff2 * (u_t - sigma_t * u_eps))
    noise = torch.randn_like(mu_denoise)
    return mu_denoise + idct_2d(torch.sqrt(torch.clamp(sigma2_denoise, min=delta)) * noise)

# --- GaussianDiffusion Class --- #
class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, context_fn, channels=3, num_timesteps=1000,
                 loss_type="l2", lagrangian=1e-3, aux_loss_weight=0, aux_loss_type="l2", vbr=False):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        self.vbr = vbr
        self.loss_fn_vgg = lpips.LPIPS(net="vgg") if aux_loss_weight > 0 else None

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_freq = dct_2d(x_start).real
        alpha, sigma = get_alpha_sigma(t, img_dim=x_start.shape[-1], channels=self.channels)
        return idct_2d(alpha * x_freq) + sigma * noise

    def p_losses(self, x_start, context_dict, t):
        noise = torch.randn_like(x_start)
        x_noisy, eps = diffuse(x_start, t)
        fx = self.denoise_fn(x_noisy, t.unsqueeze(-1), context=context_dict["output"])

        if self.loss_type == "l1":
            err = F.l1_loss(eps, fx)
        elif self.loss_type == "l2":
            err = F.mse_loss(eps, fx)
        else:
            raise NotImplementedError()

        aux_err = 0
        if self.aux_loss_weight > 0:
            pred_x0 = x_start  # replace with actual reconstruction
            if self.aux_loss_type == "l1":
                aux_err = F.l1_loss(x_start, pred_x0)
            elif self.aux_loss_type == "l2":
                aux_err = F.mse_loss(x_start, pred_x0)
            elif self.aux_loss_type == "lpips":
                aux_err = self.loss_fn_vgg(x_start, pred_x0).mean()
            else:
                raise NotImplementedError()

            loss = self.lagrangian_beta * context_dict["bpp"].mean() + err * (1 - self.aux_loss_weight) + aux_err * self.aux_loss_weight
        else:
            loss = self.lagrangian_beta * context_dict["bpp"].mean() + err
        return loss

    def forward(self, x_start):
        t = torch.rand(x_start.size(0), device=x_start.device)
        bitrate_scale = torch.rand(x_start.size(0), device=x_start.device) if self.vbr else None
        if self.vbr:
            self.lagrangian_beta = self.scale_to_beta(bitrate_scale)
        context_dict = self.context_fn(x_start, bitrate_scale)
        loss = self.p_losses(x_start, context_dict, t)
        extra = self.context_fn.get_extra_loss() if hasattr(self.context_fn, "get_extra_loss") else None
        return loss, extra

    def scale_to_beta(self, bitrate_scale):
        return 2 ** (3 * bitrate_scale) * 5e-4

    @torch.no_grad()
    def p_sample_loop(self, shape, context, init=None):
        img = torch.zeros(shape, device=next(self.parameters()).device) if init is None else init
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i / self.num_timesteps, device=img.device)
            img = denoise(img, t, self.num_timesteps,
                          lambda x, t_: self.denoise_fn(x, t_.unsqueeze(-1), context=context),
                          img_dim=shape[-1])
        return img

    @torch.no_grad()
    def compress(self, images, sample_steps=None, bitrate_scale=None, sample_mode="ddpm", init=None):
        context_dict = self.context_fn(images, bitrate_scale)
        T = self.num_timesteps if sample_steps is None else sample_steps
        samples = self.p_sample_loop(images.shape, context_dict["output"], init=init)
        return samples, context_dict["bpp"].mean()

