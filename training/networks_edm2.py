# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""EDM2 network donor for the ImageNet-64 ECM recipe."""

import numpy as np
import torch

from torch_utils import misc
from torch_utils import persistence


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(
        x, dim=dim, keepdim=True, dtype=torch.float32
    )
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def resample(x, f=(1, 1), mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    if f.ndim != 1 or len(f) % 2 != 0:
        raise ValueError('resampling filter must have positive even length')
    padding = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    channels = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(
            x, f.tile([channels, 1, 1, 1]), groups=channels,
            stride=2, padding=(padding,),
        )
    if mode != 'up':
        raise ValueError(f'unsupported resampling mode: {mode!r}')
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([channels, 1, 1, 1]), groups=channels,
        stride=2, padding=(padding,),
    )


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a, b, dim=1, t=0.5):
    a_channels = a.shape[dim]
    b_channels = b.shape[dim]
    scale = np.sqrt(
        (a_channels + b_channels) / ((1 - t) ** 2 + t ** 2)
    )
    a_weight = scale / np.sqrt(a_channels) * (1 - t)
    b_weight = scale / np.sqrt(b_channels) * t
    return torch.cat([a_weight * a, b_weight * b], dim=dim)


@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer(
            'freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth
        )
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32).ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        return (y.cos() * np.sqrt(2)).to(x.dtype)


@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )
        self.force_wn = True

    def forward(self, x, gain=1):
        weight = self.weight.to(torch.float32)
        if self.training and self.force_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(weight))
        weight = normalize(weight)
        weight = weight * (gain / np.sqrt(weight[0].numel()))
        weight = weight.to(x.dtype)
        if weight.ndim == 2:
            return x @ weight.t()
        return torch.nn.functional.conv2d(
            x, weight, padding=(weight.shape[-1] // 2,)
        )


@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        flavor='enc',
        resample_mode='keep',
        resample_filter=(1, 1),
        attention=False,
        channels_per_head=64,
        dropout=0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = (
            out_channels // channels_per_head if attention else 0
        )
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == 'enc' else in_channels,
            out_channels,
            kernel=[3, 3],
        )
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads else None
        )

    def forward(self, x, emb):
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        y = self.conv_res0(mp_silu(x))
        gain = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * gain.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads:
            y = self.attn_qkv(x)
            y = y.reshape(
                y.shape[0], self.num_heads, -1, 3,
                y.shape[2] * y.shape[3],
            )
            query, key, value = normalize(y, dim=2).unbind(3)
            weights = torch.einsum(
                'nhcq,nhck->nhqk',
                query,
                key / np.sqrt(query.shape[2]),
            ).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', weights, value)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim,
        model_channels=192,
        channel_mult=(1, 2, 3, 4),
        channel_mult_noise=None,
        channel_mult_emb=None,
        num_blocks=3,
        attn_resolutions=(16, 8),
        label_balance=0.5,
        concat_balance=0.5,
        dropout_res=None,
        **block_kwargs,
    ):
        super().__init__()
        block_channels = [model_channels * value for value in channel_mult]
        noise_channels = (
            model_channels * channel_mult_noise
            if channel_mult_noise is not None else block_channels[0]
        )
        emb_channels = (
            model_channels * channel_mult_emb
            if channel_mult_emb is not None else max(block_channels)
        )
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_fourier = MPFourier(noise_channels)
        self.emb_noise = MPConv(noise_channels, emb_channels, kernel=[])
        self.emb_label = (
            MPConv(label_dim, emb_channels, kernel=[])
            if label_dim else None
        )

        self.enc = torch.nn.ModuleDict()
        out_channels = img_channels + 1
        for level, channels in enumerate(block_channels):
            resolution = img_resolution >> level
            kwargs = block_kwargs.copy()
            if dropout_res and resolution > dropout_res:
                kwargs['dropout'] = 0.0
            if level == 0:
                in_channels = out_channels
                out_channels = channels
                self.enc[f'{resolution}x{resolution}_conv'] = MPConv(
                    in_channels, out_channels, kernel=[3, 3]
                )
            else:
                self.enc[f'{resolution}x{resolution}_down'] = Block(
                    out_channels, out_channels, emb_channels, flavor='enc',
                    resample_mode='down', **kwargs,
                )
            for index in range(num_blocks):
                in_channels = out_channels
                out_channels = channels
                self.enc[f'{resolution}x{resolution}_block{index}'] = Block(
                    in_channels, out_channels, emb_channels, flavor='enc',
                    attention=(resolution in attn_resolutions), **kwargs,
                )

        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(block_channels))):
            resolution = img_resolution >> level
            kwargs = block_kwargs.copy()
            if dropout_res and resolution > dropout_res:
                kwargs['dropout'] = 0.0
            if level == len(block_channels) - 1:
                self.dec[f'{resolution}x{resolution}_in0'] = Block(
                    out_channels, out_channels, emb_channels, flavor='dec',
                    attention=True, **kwargs,
                )
                self.dec[f'{resolution}x{resolution}_in1'] = Block(
                    out_channels, out_channels, emb_channels, flavor='dec',
                    **kwargs,
                )
            else:
                self.dec[f'{resolution}x{resolution}_up'] = Block(
                    out_channels, out_channels, emb_channels, flavor='dec',
                    resample_mode='up', **kwargs,
                )
            for index in range(num_blocks + 1):
                in_channels = out_channels + skips.pop()
                out_channels = channels
                self.dec[f'{resolution}x{resolution}_block{index}'] = Block(
                    in_channels, out_channels, emb_channels, flavor='dec',
                    attention=(resolution in attn_resolutions), **kwargs,
                )
        self.out_conv = MPConv(out_channels, img_channels, kernel=[3, 3])

    def forward(self, x, noise_labels, class_labels, **_kwargs):
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            label_emb = self.emb_label(
                class_labels * np.sqrt(class_labels.shape[1])
            )
            emb = mp_sum(emb, label_emb, t=self.label_balance)
        emb = mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        return self.out_conv(x, gain=self.out_gain)


@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim,
        use_fp16=True,
        sigma_data=0.5,
        logvar_channels=128,
        **unet_kwargs,
    ):
        super().__init__()
        del logvar_channels
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        unet_kwargs.pop('augment_dim', None)
        self.unet = UNet(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
            **unet_kwargs,
        )

    def forward(
        self, x, sigma, class_labels=None, force_fp32=False,
        return_logvar=False, **unet_kwargs,
    ):
        del return_logvar
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros(
                [1, self.label_dim], device=x.device
            )
        else:
            class_labels = class_labels.to(torch.float32).reshape(
                -1, self.label_dim
            )
        dtype = (
            torch.float16
            if self.use_fp16 and not force_fp32 and x.device.type == 'cuda'
            else torch.float32
        )
        c_skip = self.sigma_data ** 2 / (
            sigma ** 2 + self.sigma_data ** 2
        )
        c_out = sigma * self.sigma_data / (
            sigma ** 2 + self.sigma_data ** 2
        ).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        model_output = self.unet(
            (c_in * x).to(dtype), c_noise, class_labels, **unet_kwargs
        )
        return c_skip * x + c_out * model_output.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
