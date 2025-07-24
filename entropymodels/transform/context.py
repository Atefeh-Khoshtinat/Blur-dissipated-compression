import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from entropymodels.utils_entropy.ckbd import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    Masked convolution where the mask follows a checkerboard pattern.
    1s at anchor positions and 0s at non-anchor positions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        self.mask[:, :, ::2, 1::2] = 0
        self.mask[:, :, 1::2, ::2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class LocalContext(nn.Module):
    """
    Replaces windowed attention with a checkerboard-masked convolution followed by an MLP.
    The output shape will be (B, 2*C, H, W) to match the original class design.
    """
    def __init__(self, dim=32, window_size=5, mlp_ratio=2.):
        super().__init__()
        self.masked_conv = CheckboardMaskedConv2d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=window_size,
            padding=window_size // 2,
            stride=1
        )
        self.norm = nn.BatchNorm2d(dim * 2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, int(dim * 2 * mlp_ratio), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(dim * 2 * mlp_ratio), dim * 2, kernel_size=1)
        )

    def forward(self, x):
        x = self.masked_conv(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x




class ChannelContext(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 192, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim * 4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fushion(channel_params)

        return channel_params
########################################################################
# === Laplacian Relative Bias ===
class LaplacianRelativeBias(nn.Module):
    def __init__(self, A=1.0, B=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))

    def forward(self, rel_coords, num_heads):
        abs_sum = torch.abs(rel_coords[..., 0]) + torch.abs(rel_coords[..., 1])
        exponent = -0.5 * (abs_sum / (self.B**2))
        bias = (self.A**2) * torch.exp(exponent)
        return bias.unsqueeze(0).unsqueeze(0).repeat(1, num_heads, 1, 1)

# === Window-based Self-Attention (Checkerboard-aware conv QKV) ===
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # older PyTorch compatible
        coords_flatten = torch.flatten(coords, 1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        self.register_buffer("rel_coords", rel_coords.permute(1, 2, 0).contiguous())

        self.lap_bias = LaplacianRelativeBias()

        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )

        self.reprojection = nn.Conv2d(dim, dim, kernel_size=5, padding=2)  # ? fixed: output is dim
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)  # ? fixed: output is dim
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B_, C, H, W)

        # Checkerboard partitioning
        mask = torch.zeros_like(x[:, :1])
        mask[:, :, 0::2, 0::2] = 1
        mask[:, :, 1::2, 1::2] = 1
        q_mask = 1 - mask

        q = self.q_proj(x * q_mask).reshape(B_, self.num_heads, C // self.num_heads, N)
        k = self.k_proj(x * mask).reshape(B_, self.num_heads, C // self.num_heads, N)
        v = self.v_proj(x * mask).reshape(B_, self.num_heads, C // self.num_heads, N)

        q = q * self.scale
        attn = q.transpose(-2, -1) @ k
        bias = self.lap_bias(self.rel_coords.to(x.device), self.num_heads)
        attn = attn + bias
        attn = self.softmax(attn)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B_, C, H, W)
        out = self.reprojection(out)
        return (out + self.mlp(out)).reshape(B_, self.window_size * self.window_size, self.dim)  # ? safe reshape

# === Window ops ===
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C, B):
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x

# === SwinBlock with checkerboard attention ===
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.shift_size > 0:
            for b in range(B):
                for h in range(0, H, self.window_size):
                    if (h // self.window_size) % 2 == 1:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], shifts=(-self.shift_size), dims=2)
                    else:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], shifts=(self.shift_size), dims=2)

        windows = window_partition(x, self.window_size)
        x = self.attn(windows)

        num_windows = (H // self.window_size) * (W // self.window_size)
        real_B = x.shape[0] // num_windows
        x = window_reverse(x, self.window_size, H, W, C, real_B)

        if self.shift_size > 0:
            for b in range(real_B):
                for h in range(0, H, self.window_size):
                    if (h // self.window_size) % 2 == 1:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], shifts=(self.shift_size), dims=2)
                    else:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], shifts=(-self.shift_size), dims=2)

        x = x.permute(0, 2, 3, 1).contiguous().view(real_B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.view(real_B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class GlobalIntraContext(nn.Module):
    def __init__(self, dim=256, depth=4, num_heads=4, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depth)
        ])
        self.reprojection = nn.Conv2d(dim,  dim*2, kernel_size=5, stride=1, padding=2)
        self.skip = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.reprojection(x) + self.skip(x)
###################################################################################
# --- Laplacian Relative Position Bias ---
class LaplacianRelativeBias(nn.Module):
    def __init__(self, A=1.0, B=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))

    def forward(self, rel_coords, num_heads):
        abs_sum = torch.abs(rel_coords[..., 0]) + torch.abs(rel_coords[..., 1])
        exponent = -0.5 * (abs_sum / (self.B**2))
        bias = (self.A**2) * torch.exp(exponent)
        return bias.unsqueeze(0).unsqueeze(0).repeat(1, num_heads, 1, 1)

# --- Window Attention with Laplacian Bias ---
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # Compatible with PyTorch <1.10
        coords_flatten = torch.flatten(coords, 1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        self.register_buffer("rel_coords", rel_coords.permute(1, 2, 0).contiguous())

        self.lap_bias = LaplacianRelativeBias()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        bias = self.lap_bias(self.rel_coords.to(x.device), self.num_heads)
        attn = attn + bias

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

# --- Partition/Reverse Utilities ---
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x

# --- SwinBlock with Laplacian Bias ---
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.shift_size > 0:
            for b in range(B):
                for h in range(0, H, self.window_size):
                    if (h // self.window_size) % 2 == 1:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], -self.shift_size, dims=2)
                    else:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], self.shift_size, dims=2)

        windows = window_partition(x, self.window_size)
        x = self.attn(windows)
        x = window_reverse(x, self.window_size, H, W, C)

        if self.shift_size > 0:
            for b in range(B):
                for h in range(0, H, self.window_size):
                    if (h // self.window_size) % 2 == 1:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], self.shift_size, dims=2)
                    else:
                        x[b, :, h:h+self.window_size] = torch.roll(x[b, :, h:h+self.window_size], -self.shift_size, dims=2)

        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

# --- GlobalInterContext (Wraps SwinBlock) ---
class GlobalInterContext(nn.Module):
    def __init__(self, dim=96, num_heads=4, window_size=8, shift_size=4):
        super().__init__()
        self.block = SwinBlock(dim, num_heads, window_size, shift_size)
        self.reprojection = nn.Conv2d(dim,  dim*2, kernel_size=5, stride=1, padding=2)
        self.skip = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        trans_out=self.block(x)
        return self.reprojection(trans_out)+self.skip(trans_out)





