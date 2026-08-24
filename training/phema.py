# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""Power-function EMA from "Analyzing and Improving Training Dynamics"."""

import copy

import numpy as np
import torch


def exp_to_std(exp):
    exp = np.float64(exp)
    return np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))


def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - value, 12 - value]).real.max() for value in tmp]
    return np.float64(exp).reshape(std.shape)


def power_function_response(ofs, std, length, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    shape = [1] * exp.ndim
    shape[axis] = -1
    t = np.arange(length).reshape(shape)
    response = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    return response / np.sum(response, axis=axis, keepdims=True)


def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    numerator = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    denominator = (a_exp + b_exp + 1) * t_max
    return numerator / denominator


def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std):
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    row = lambda value: np.float64(value).reshape(-1, 1)
    column = lambda value: np.float64(value).reshape(1, -1)
    matrix = power_function_correlation(
        row(in_ofs), row(in_std), column(in_ofs), column(in_std)
    )
    target = power_function_correlation(
        row(in_ofs), row(in_std), column(out_ofs), column(out_std)
    )
    coefficients = np.linalg.solve(matrix, target)
    return coefficients / np.sum(coefficients, axis=0)


def power_function_beta(std, t_next, t_delta):
    return (1 - t_delta / t_next) ** (std_to_exp(std) + 1)


class PowerFunctionEMA:
    @torch.no_grad()
    def __init__(self, net, stds=(0.010, 0.050, 0.100)):
        self.net = net
        self.stds = tuple(float(std) for std in stds)
        if not self.stds or any(std <= 0 for std in self.stds):
            raise ValueError('PowerEMA stds must be positive')
        self.emas = [copy.deepcopy(net).eval().requires_grad_(False) for _ in self.stds]

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            ema.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        if cur_nimg < batch_size or batch_size <= 0:
            raise ValueError('PowerEMA update requires cur_nimg >= batch_size > 0')
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(
                std=std, t_next=cur_nimg, t_delta=batch_size
            )
            for net_param, ema_param in zip(
                self.net.parameters(), ema.parameters()
            ):
                ema_param.lerp_(net_param, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for net_buffer, ema_buffer in zip(
                self.net.buffers(), ema.buffers()
            ):
                ema_buffer.copy_(net_buffer)
        return [
            (ema, f'-{std:.3f}')
            for std, ema in zip(self.stds, self.emas)
        ]

    def state_dict(self):
        return {
            'stds': self.stds,
            'emas': [ema.state_dict() for ema in self.emas],
        }

    def load_state_dict(self, state):
        saved_stds = tuple(float(std) for std in state['stds'])
        saved_emas = state['emas']
        if saved_stds != self.stds or len(saved_emas) != len(self.emas):
            raise ValueError('PowerEMA profiles do not match current config')
        for ema, ema_state in zip(self.emas, saved_emas):
            ema.load_state_dict(ema_state)

