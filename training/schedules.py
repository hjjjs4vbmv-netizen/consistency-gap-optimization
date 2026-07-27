"""t -> r mapping schedules for Easy Consistency Tuning (ECT).

During consistency tuning, every training pair (x_t, x_r) is built from a
noise level t ~ p(t) and a smaller noise level r = r(t, stage) produced by a
mapping schedule ("Consistency Models Made Easy", arXiv 2406.14548, Sec. 3.3
and Appendix A). This module centralizes the t -> r schedules behind a single
interface; ECMLoss in training/loss.py dispatches its t -> r entry through it
as r = self.schedule.compute_r(t=t, stage=self.stage), while the official
reference formulas stay verbatim in training/loss.py as the parity anchor.

Supported schedules:
    'const'        Official ECT constant mapping, Eq. (17).
    'sigmoid'      Official ECT sigmoid mapping, Eq. (18); training default.
    'global_sigmoid' Official sigmoid gap times one fixed global multiplier.
    'adaptive_v1'  Official sigmoid ratio plus a bounded correction driven by
                   the EMA of the globally aggregated training loss.
    'local_tbin_v1' Official sigmoid gap times a bounded per-t-bin multiplier
                    driven by unweighted raw pair-loss trends.
    'local_tbin_v2' V1 signal with conservative bounds and equal-bin
                    geometric-mean normalization that preserves global scale.
    'local_tbin_v3' V2 local redistribution times an explicit global gap
                    multiplier, separating calibration from adaptation.

The 'const' and 'sigmoid' formulas are verbatim ports of
ECMLoss.t_to_r_const / ECMLoss.t_to_r_sigmoid in training/loss.py and MUST NOT
be modified: they are the official fixed baseline this project reproduces.
tests/test_schedules.py enforces bitwise parity against training/loss.py.

Usage (both forms are supported):
    from training.schedules import compute_r, get_schedule

    schedule = get_schedule('sigmoid', q=256, k=8, b=1)
    r = schedule.compute_r(t=t, stage=stage)

    r = compute_r(t=t, stage=stage, schedule='sigmoid', q=256, k=8, b=1)

`t` may be a torch tensor of any shape (the training loop uses [N, 1, 1, 1]),
or a python/numpy scalar or array, which is converted via torch.as_tensor();
the result is a tensor of the same shape with r clamped to r >= 0. `stage` is
the official integer curriculum stage maintained by the training loop
(stage = cur_tick // double_ticks). adaptive_v1 changes only r/t using the
loss EMA; it does not replace the official stage curriculum.
"""

import math
from statistics import NormalDist

import torch

#----------------------------------------------------------------------------
# Registry.

_SCHEDULES = {}

def register_schedule(name):
    def decorator(cls):
        cls.name = name
        _SCHEDULES[name] = cls
        return cls
    return decorator

def available_schedules():
    return sorted(_SCHEDULES)

def get_schedule(schedule, **schedule_kwargs):
    # training/loss.py imports this by name, and torch_utils.persistence
    # embeds that module's source into training snapshots — keep the public
    # names in this module stable or old snapshots stop unpickling.
    if schedule not in _SCHEDULES:
        raise ValueError(f"Unknown schedule type {schedule!r}! Available: {', '.join(available_schedules())}")
    return _SCHEDULES[schedule](**schedule_kwargs)

#----------------------------------------------------------------------------
# Interface. Hyperparameter defaults follow ct_train.py (-q 2.0 -k 8.0 -b 1.0).

class Schedule:
    name = None

    def __init__(self, q=2.0, k=8.0, b=1.0):
        if q <= 1:
            raise ValueError(f'q must be > 1 (Delta_t decay factor), got {q}')
        self.q = q
        self.k = k
        self.b = b
        self.stage = 0

    def compute_r(self, t, stage):
        raise NotImplementedError

    # Stateful interface mirroring ECMLoss, so a Schedule instance can drive
    # the existing training loop (update_schedule() at stage boundaries,
    # t_to_r() inside the loss) without further changes.
    def update_schedule(self, stage):
        self.stage = stage

    def t_to_r(self, t):
        return self.compute_r(t=t, stage=self.stage)

    def update_training_signal(self, loss):
        del loss
        return False

    def runtime_metrics(self):
        """Stable controller telemetry contract for training/evaluation code."""
        return {
            'loss_ema': None,
            'loss_reference': None,
            'correction': 0.0,
            'signal_updates': 0,
            'adaptive_active': False,
        }

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError(f'{type(self).__name__} does not have adaptive state')

    def metadata(self):
        return {
            'name': self.name,
            'enabled': False,
            'q': self.q,
            'k': self.k,
            'b': self.b,
        }

    def __repr__(self):
        return f'{type(self).__name__}(q={self.q}, k={self.k}, b={self.b})'

def _as_tensor(t):
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)

#----------------------------------------------------------------------------
# Official fixed schedules. Verbatim ports of training/loss.py — do not edit
# the formulas; tests/test_schedules.py checks them bit-for-bit against
# ECMLoss.

@register_schedule('const')
class ConstSchedule(Schedule):
    """Official constant mapping, Eq. (17): r/t = 1 - 1/q^(stage+1).

    Port of ECMLoss.t_to_r_const in training/loss.py.
    """

    def compute_r(self, t, stage):
        t = _as_tensor(t)
        decay = 1 / self.q ** (stage + 1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

@register_schedule('sigmoid')
class SigmoidSchedule(Schedule):
    """Official sigmoid mapping, Eq. (18): r/t = 1 - n(t)/q^(stage+1), where
    n(t) = 1 + k * sigmoid(-b * t). Training default (--mapping=sigmoid).

    Port of ECMLoss.t_to_r_sigmoid in training/loss.py.
    """

    def compute_r(self, t, stage):
        t = _as_tensor(t)
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

#----------------------------------------------------------------------------
# Experimental schedules (Role C). Changes relative to the official fixed
# schedules live below this line only.

def _validate_global_gap_scale(global_gap_scale):
    value = float(global_gap_scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f'global_gap_scale must be finite and > 0, got {global_gap_scale}'
        )
    return value


def _apply_global_gap_scale(t, base_r, global_gap_scale):
    """Scale ``t - base_r`` while preserving ``0 <= r <= t``.

    The exact ``scale == 1`` branch protects bitwise parity with the schedule
    being wrapped, which makes the factorized controls auditable.
    """
    if global_gap_scale == 1.0:
        return base_r
    t = _as_tensor(t)
    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
        base_r = base_r.to(t.dtype)
    finite_max = torch.finfo(t.dtype).max
    safe_t = torch.nan_to_num(
        t, nan=0.0, posinf=finite_max, neginf=0.0
    ).clamp_min(0)
    safe_r = torch.nan_to_num(
        base_r, nan=0.0, posinf=finite_max, neginf=0.0
    ).clamp_min(0)
    base_gap = (safe_t - torch.minimum(safe_r, safe_t)).clamp_min(0)
    scaled_gap = base_gap * global_gap_scale
    scaled_gap = torch.minimum(
        torch.nan_to_num(
            scaled_gap, nan=0.0, posinf=finite_max, neginf=0.0
        ).clamp_min(0),
        safe_t,
    )
    return safe_t - scaled_gap


@register_schedule('global_sigmoid')
class GlobalSigmoidSchedule(SigmoidSchedule):
    """Official sigmoid mapping with one fixed multiplier on every gap."""

    def __init__(self, q=2.0, k=8.0, b=1.0, global_gap_scale=1.0):
        super().__init__(q=q, k=k, b=b)
        self.global_gap_scale = _validate_global_gap_scale(global_gap_scale)

    def compute_r(self, t, stage):
        base_r = super().compute_r(t=t, stage=stage)
        return _apply_global_gap_scale(t, base_r, self.global_gap_scale)

    def runtime_metrics(self):
        metrics = super().runtime_metrics()
        metrics['correction'] = self.global_gap_scale - 1.0
        return metrics

    def metadata(self):
        metadata = super().metadata()
        metadata.update(
            name=self.name,
            enabled=True,
            intervention='fixed_global_multiplier_on_official_sigmoid_gap',
            global_gap_scale=self.global_gap_scale,
            **self.runtime_metrics(),
        )
        return metadata


@register_schedule('adaptive_v1')
class AdaptiveV1Schedule(SigmoidSchedule):
    """Loss-EMA adaptive correction on top of the official sigmoid ratio.

    Let rho_0 be the official sigmoid r/t ratio, L_ref the loss EMA at the
    end of warm-up, and L_ema the current loss EMA. The correction is

        delta = max_adjust * tanh(log(L_ref) - log(L_ema))

    and rho = clamp(rho_0 + delta, 0, 1 - min_gap). Improving loss therefore
    tightens the pair (smaller t-r), while worsening loss widens it. The
    correction is deterministic and bounded by max_adjust.
    """

    def __init__(self, q=2.0, k=8.0, b=1.0, loss_ema_beta=0.9,
                 max_adjust=0.05, min_gap=1e-3, warmup_updates=2):
        super().__init__(q=q, k=k, b=b)
        for name, value in [('q', q), ('k', k), ('b', b)]:
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite, got {value}')
        if not math.isfinite(loss_ema_beta) or not 0 <= loss_ema_beta < 1:
            raise ValueError(f'loss_ema_beta must be in [0, 1), got {loss_ema_beta}')
        if not math.isfinite(max_adjust) or not 0 <= max_adjust <= 1:
            raise ValueError(f'max_adjust must be in [0, 1], got {max_adjust}')
        if not math.isfinite(min_gap) or not 0 < min_gap < 1:
            raise ValueError(f'min_gap must be in (0, 1), got {min_gap}')
        try:
            normalized_warmup_updates = int(warmup_updates)
        except (TypeError, ValueError, OverflowError):
            normalized_warmup_updates = -1
        if (isinstance(warmup_updates, bool) or normalized_warmup_updates != warmup_updates
                or normalized_warmup_updates < 0):
            raise ValueError(f'warmup_updates must be a non-negative integer, got {warmup_updates}')
        self.loss_ema_beta = float(loss_ema_beta)
        self.max_adjust = float(max_adjust)
        self.min_gap = float(min_gap)
        self.warmup_updates = normalized_warmup_updates
        self.loss_ema = None
        self.loss_reference = None
        self.signal_updates = 0

    def update_training_signal(self, loss):
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            return False
        loss = max(loss, torch.finfo(torch.float64).tiny)
        if self.loss_ema is None:
            updated_ema = loss
        else:
            beta = self.loss_ema_beta
            updated_ema = beta * self.loss_ema + (1 - beta) * loss
        if not math.isfinite(updated_ema) or updated_ema <= 0:
            return False
        self.loss_ema = updated_ema
        self.signal_updates += 1

        # Establish the baseline only after the requested number of valid
        # signals have contributed to the EMA. With no warm-up, the first
        # signal is necessarily the baseline (and therefore has zero
        # correction); otherwise the following signal is the first one that
        # can produce a correction relative to this reference.
        if self.loss_reference is None and (
            self.warmup_updates == 0 or self.signal_updates == self.warmup_updates
        ):
            self.loss_reference = updated_ema
        return True

    def correction_is_active(self):
        return (
            self.max_adjust != 0
            and self.loss_ema is not None
            and self.loss_reference is not None
            and self.signal_updates > self.warmup_updates
        )

    def correction(self):
        if not self.correction_is_active():
            return 0.0
        log_improvement = math.log(self.loss_reference) - math.log(self.loss_ema)
        return self.max_adjust * math.tanh(log_improvement)

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)

        # Before a correction is active, adaptive_v1 is exactly the official
        # sigmoid schedule. In particular, min_gap must not alter the no-signal
        # or warmup path.
        if not self.correction_is_active():
            return super().compute_r(t=t, stage=stage)

        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        finite_max = torch.finfo(t.dtype).max
        t = torch.nan_to_num(t, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0)

        try:
            base_r = super().compute_r(t=t, stage=stage)
        except OverflowError:
            # q**(stage+1) -> inf, so the mathematical sigmoid ratio -> 1.
            base_r = t
        delta = self.correction()
        base_ratio = torch.where(t > 0, base_r / t, torch.zeros_like(t))
        ratio = torch.clamp(base_ratio + delta, min=0, max=1 - self.min_gap)
        r = torch.nan_to_num(t * ratio, nan=0.0, posinf=finite_max, neginf=0.0)
        return torch.minimum(r.clamp_min(0), t)

    def state_dict(self):
        return {
            'loss_ema': self.loss_ema,
            'loss_reference': self.loss_reference,
            'signal_updates': self.signal_updates,
        }

    def load_state_dict(self, state):
        loss_ema = state.get('loss_ema')
        loss_reference = state.get('loss_reference')
        signal_updates = int(state.get('signal_updates', 0))
        for name, value in [('loss_ema', loss_ema), ('loss_reference', loss_reference)]:
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        if signal_updates < 0:
            raise ValueError(f'signal_updates must be >= 0, got {signal_updates}')
        self.loss_ema = None if loss_ema is None else float(loss_ema)
        self.loss_reference = None if loss_reference is None else float(loss_reference)
        self.signal_updates = signal_updates

    def metadata(self):
        return {
            'name': self.name,
            'enabled': True,
            'signal': 'loss_ema',
            'q': self.q,
            'k': self.k,
            'b': self.b,
            'loss_ema_beta': self.loss_ema_beta,
            'warmup_updates': self.warmup_updates,
            'max_adjust': self.max_adjust,
            'min_gap': self.min_gap,
            **self.runtime_metrics(),
        }

    def runtime_metrics(self):
        return {
            'loss_ema': self.loss_ema,
            'loss_reference': self.loss_reference,
            'correction': self.correction(),
            'signal_updates': self.signal_updates,
            'adaptive_active': self.correction_is_active(),
        }


@register_schedule('local_tbin_v1')
class LocalTBinV1Schedule(SigmoidSchedule):
    """Conservative local controller on top of the official sigmoid gap.

    The log-normal training distribution is divided into equal-probability
    bins.  Each bin tracks short and long EMAs of the *raw squared pair loss*
    supplied by :class:`ECMLoss`.  The official sigmoid mapping remains the
    baseline and only its relative gap is scaled:

        base_gap = (t - r_sigmoid) / t
        trend_j = tanh(log(long_ema_j) - log(short_ema_j))
        scale_j = clip(exp(-gain * trend_j), min_scale, max_scale)
        r = t * (1 - clip(base_gap * scale_j, min_gap, 1))

    Falling short-term loss tightens the local pair (scale < 1); rising loss
    widens it toward the diffusion-pretraining boundary (scale > 1).  A
    two-timescale trend avoids freezing a reference during the startup
    transient, and the multiplicative bound keeps the correction relative to
    the official t-dependent gap instead of replacing its n(t) structure.
    """

    def __init__(self, q=2.0, k=8.0, b=1.0, p_mean=-1.1, p_std=2.0,
                 num_bins=4, short_beta=0.9, long_beta=0.99,
                 warmup_updates=32, gain=0.5, min_scale=0.75,
                 max_scale=1.5, deadband=0.02, min_gap=1e-3):
        super().__init__(q=q, k=k, b=b)
        if not math.isfinite(float(p_mean)):
            raise ValueError(f'p_mean must be finite, got {p_mean}')
        if not math.isfinite(float(p_std)) or float(p_std) <= 0:
            raise ValueError(f'p_std must be finite and > 0, got {p_std}')
        if isinstance(num_bins, bool) or int(num_bins) != num_bins or int(num_bins) < 2:
            raise ValueError(f'num_bins must be an integer >= 2, got {num_bins}')
        for name, value in [('short_beta', short_beta), ('long_beta', long_beta)]:
            if not math.isfinite(float(value)) or not 0 <= float(value) < 1:
                raise ValueError(f'{name} must be in [0, 1), got {value}')
        if float(short_beta) >= float(long_beta):
            raise ValueError('short_beta must be smaller than long_beta')
        if (isinstance(warmup_updates, bool) or int(warmup_updates) != warmup_updates
                or int(warmup_updates) < 0):
            raise ValueError(
                f'warmup_updates must be a non-negative integer, got {warmup_updates}'
            )
        for name, value in [('gain', gain), ('deadband', deadband), ('min_gap', min_gap)]:
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f'{name} must be finite and >= 0, got {value}')
        if not 0 < float(min_gap) < 1:
            raise ValueError(f'min_gap must be in (0, 1), got {min_gap}')
        if not math.isfinite(float(min_scale)) or not 0 < float(min_scale) <= 1:
            raise ValueError(f'min_scale must be in (0, 1], got {min_scale}')
        if not math.isfinite(float(max_scale)) or float(max_scale) < 1:
            raise ValueError(f'max_scale must be finite and >= 1, got {max_scale}')

        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.num_bins = int(num_bins)
        self.short_beta = float(short_beta)
        self.long_beta = float(long_beta)
        self.warmup_updates = int(warmup_updates)
        self.gain = float(gain)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.deadband = float(deadband)
        self.min_gap = float(min_gap)
        normal = NormalDist()
        self.log_bin_edges = [
            self.p_mean + self.p_std * normal.inv_cdf(index / self.num_bins)
            for index in range(1, self.num_bins)
        ]
        self.short_ema = [None] * self.num_bins
        self.long_ema = [None] * self.num_bins
        self.last_raw_loss = [None] * self.num_bins
        self.bin_updates = [0] * self.num_bins

    def bin_indices(self, t):
        """Return p(t)-quantile bin indices with the same shape as ``t``."""
        t = _as_tensor(t)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        tiny = torch.finfo(t.dtype).tiny
        log_t = torch.log(torch.nan_to_num(t, nan=tiny, posinf=torch.finfo(t.dtype).max,
                                           neginf=tiny).clamp_min(tiny))
        boundaries = torch.tensor(self.log_bin_edges, dtype=t.dtype, device=t.device)
        return torch.bucketize(log_t, boundaries)

    def update_training_signal(self, raw_bin_losses):
        try:
            values = list(raw_bin_losses)
        except TypeError as exc:
            raise ValueError('local_tbin_v1 expects one raw loss mean per bin') from exc
        if len(values) != self.num_bins:
            raise ValueError(
                f'expected {self.num_bins} raw bin losses, got {len(values)}'
            )
        updated_any = False
        for index, value in enumerate(values):
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value < 0:
                continue
            value = max(value, torch.finfo(torch.float64).tiny)
            if self.short_ema[index] is None:
                short = value
                long = value
            else:
                short = (
                    self.short_beta * self.short_ema[index]
                    + (1 - self.short_beta) * value
                )
                long = (
                    self.long_beta * self.long_ema[index]
                    + (1 - self.long_beta) * value
                )
            if not all(math.isfinite(item) and item > 0 for item in (short, long)):
                continue
            self.short_ema[index] = short
            self.long_ema[index] = long
            self.last_raw_loss[index] = value
            self.bin_updates[index] += 1
            updated_any = True
        return updated_any

    def bin_is_active(self, index):
        return (
            self.gain != 0
            and self.short_ema[index] is not None
            and self.long_ema[index] is not None
            and self.bin_updates[index] > self.warmup_updates
        )

    def gap_scales(self):
        scales = []
        for index in range(self.num_bins):
            if not self.bin_is_active(index):
                scales.append(1.0)
                continue
            log_trend = math.log(self.long_ema[index]) - math.log(self.short_ema[index])
            if abs(log_trend) <= self.deadband:
                log_trend = 0.0
            score = math.tanh(log_trend)
            scale = math.exp(-self.gain * score)
            scales.append(min(max(scale, self.min_scale), self.max_scale))
        return scales

    def correction_is_active(self):
        return all(self.bin_is_active(index) for index in range(self.num_bins))

    def correction(self):
        scales = self.gap_scales()
        return sum(scale - 1 for scale in scales) / self.num_bins

    def compute_r(self, t, stage):
        stage = float(stage)
        if not math.isfinite(stage) or stage < 0:
            raise ValueError(f'stage must be finite and >= 0, got {stage}')
        t = _as_tensor(t)
        base_r = super().compute_r(t=t, stage=stage)
        if not any(self.bin_is_active(index) for index in range(self.num_bins)):
            return base_r
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
            base_r = base_r.to(t.dtype)
        finite_max = torch.finfo(t.dtype).max
        safe_t = torch.nan_to_num(t, nan=0.0, posinf=finite_max, neginf=0.0).clamp_min(0)
        bin_ids = self.bin_indices(safe_t)
        scales = torch.tensor(self.gap_scales(), dtype=t.dtype, device=t.device)[bin_ids]
        base_gap = torch.where(safe_t > 0, (safe_t - base_r) / safe_t, torch.ones_like(safe_t))
        gap = torch.clamp(base_gap * scales, min=self.min_gap, max=1.0)
        r = torch.nan_to_num(safe_t * (1 - gap), nan=0.0, posinf=finite_max, neginf=0.0)
        return torch.minimum(r.clamp_min(0), safe_t)

    def state_dict(self):
        return {
            'short_ema': self.short_ema,
            'long_ema': self.long_ema,
            'last_raw_loss': self.last_raw_loss,
            'bin_updates': self.bin_updates,
        }

    def load_state_dict(self, state):
        short = list(state.get('short_ema', [None] * self.num_bins))
        long = list(state.get('long_ema', [None] * self.num_bins))
        last = list(state.get('last_raw_loss', [None] * self.num_bins))
        updates = list(state.get('bin_updates', [0] * self.num_bins))
        if not all(len(values) == self.num_bins for values in (short, long, last, updates)):
            raise ValueError('local_tbin_v1 state bin count mismatch')
        for name, values in [('short_ema', short), ('long_ema', long), ('last_raw_loss', last)]:
            for value in values:
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError(f'{name} values must be finite and > 0')
        if any(isinstance(value, bool) or int(value) != value or int(value) < 0 for value in updates):
            raise ValueError('bin_updates values must be non-negative integers')
        self.short_ema = [None if value is None else float(value) for value in short]
        self.long_ema = [None if value is None else float(value) for value in long]
        self.last_raw_loss = [None if value is None else float(value) for value in last]
        self.bin_updates = [int(value) for value in updates]

    def local_runtime_metrics(self):
        return {
            'log_bin_edges': list(self.log_bin_edges),
            'short_ema': list(self.short_ema),
            'long_ema': list(self.long_ema),
            'last_raw_loss': list(self.last_raw_loss),
            'gap_scales': self.gap_scales(),
            'bin_updates': list(self.bin_updates),
            'bin_active': [self.bin_is_active(index) for index in range(self.num_bins)],
        }

    def runtime_metrics(self):
        short = [value for value in self.short_ema if value is not None]
        long = [value for value in self.long_ema if value is not None]
        return {
            'loss_ema': sum(short) / len(short) if short else None,
            'loss_reference': sum(long) / len(long) if long else None,
            'correction': self.correction(),
            'signal_updates': min(self.bin_updates),
            'adaptive_active': self.correction_is_active(),
        }

    def metadata(self):
        return {
            'name': self.name,
            'enabled': True,
            'signal': 'raw_pair_loss_per_quantile_t_bin',
            'q': self.q,
            'k': self.k,
            'b': self.b,
            'p_mean': self.p_mean,
            'p_std': self.p_std,
            'num_bins': self.num_bins,
            'log_bin_edges': list(self.log_bin_edges),
            'short_beta': self.short_beta,
            'long_beta': self.long_beta,
            'warmup_updates': self.warmup_updates,
            'gain': self.gain,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'deadband': self.deadband,
            'min_gap': self.min_gap,
            **self.runtime_metrics(),
            'local_metrics': self.local_runtime_metrics(),
        }


@register_schedule('local_tbin_v2')
class LocalTBinV2Schedule(LocalTBinV1Schedule):
    """Globally neutral, lower-authority local t-bin controller.

    V2 keeps V1's raw-loss signal and official sigmoid baseline, but projects
    the active log gap scales onto a bounded zero-mean set.  Since the bins are
    equal-probability under p(t), this makes their geometric mean exactly one:
    the controller redistributes gap across t regions without globally
    tightening or widening the official curriculum.
    """

    def __init__(self, q=2.0, k=8.0, b=1.0, p_mean=-1.1, p_std=2.0,
                 num_bins=4, short_beta=0.9, long_beta=0.99,
                 warmup_updates=64, gain=0.25, min_scale=0.85,
                 max_scale=1.25, deadband=0.02, min_gap=1e-3):
        super().__init__(
            q=q,
            k=k,
            b=b,
            p_mean=p_mean,
            p_std=p_std,
            num_bins=num_bins,
            short_beta=short_beta,
            long_beta=long_beta,
            warmup_updates=warmup_updates,
            gain=gain,
            min_scale=min_scale,
            max_scale=max_scale,
            deadband=deadband,
            min_gap=min_gap,
        )

    def compute_r(self, t, stage):
        # V2/V3 must remain exactly on the official sigmoid schedule until all
        # bins are ready.  Calling V1's implementation during partial warmup
        # would apply its minimum-gap clamp even though gap_scales() is neutral.
        if not self.correction_is_active():
            return SigmoidSchedule.compute_r(self, t=t, stage=stage)
        return super().compute_r(t=t, stage=stage)

    def gap_scales(self):
        # Do not partially redistribute the global curriculum while any bin is
        # still warming up. Quantile bins normally activate together, but this
        # also makes sparse or resumed signals deterministic.
        if not self.correction_is_active():
            return [1.0] * self.num_bins

        raw_scales = super().gap_scales()
        raw_logs = [math.log(scale) for scale in raw_scales]
        lower = math.log(self.min_scale)
        upper = math.log(self.max_scale)

        # Project log_scales - shift onto [lower, upper] with mean exactly zero.
        # The root exists because lower <= 0 <= upper.
        shift_low = min(value - upper for value in raw_logs)
        shift_high = max(value - lower for value in raw_logs)
        for _ in range(80):
            shift = (shift_low + shift_high) / 2
            projected = [
                min(max(value - shift, lower), upper)
                for value in raw_logs
            ]
            if sum(projected) > 0:
                shift_low = shift
            else:
                shift_high = shift
        shift = (shift_low + shift_high) / 2
        projected = [
            min(max(value - shift, lower), upper)
            for value in raw_logs
        ]
        return [math.exp(value) for value in projected]

    def local_runtime_metrics(self):
        metrics = super().local_runtime_metrics()
        scales = metrics['gap_scales']
        metrics['log_scale_mean'] = sum(math.log(scale) for scale in scales) / len(scales)
        return metrics

    def metadata(self):
        metadata = super().metadata()
        metadata.update(
            scale_normalization='equal_probability_bin_geometric_mean_1',
            global_log_scale_mean=(
                sum(math.log(scale) for scale in self.gap_scales()) / self.num_bins
            ),
        )
        return metadata


@register_schedule('local_tbin_v3')
class LocalTBinV3Schedule(LocalTBinV2Schedule):
    """Factorized global calibration and geometrically neutral local control."""

    def __init__(self, q=2.0, k=8.0, b=1.0, p_mean=-1.1, p_std=2.0,
                 num_bins=4, short_beta=0.9, long_beta=0.99,
                 warmup_updates=64, gain=0.25, min_scale=0.85,
                 max_scale=1.25, deadband=0.02, min_gap=1e-3,
                 global_gap_scale=1.0):
        super().__init__(
            q=q,
            k=k,
            b=b,
            p_mean=p_mean,
            p_std=p_std,
            num_bins=num_bins,
            short_beta=short_beta,
            long_beta=long_beta,
            warmup_updates=warmup_updates,
            gain=gain,
            min_scale=min_scale,
            max_scale=max_scale,
            deadband=deadband,
            min_gap=min_gap,
        )
        self.global_gap_scale = _validate_global_gap_scale(global_gap_scale)

    def compute_r(self, t, stage):
        local_r = super().compute_r(t=t, stage=stage)
        return _apply_global_gap_scale(t, local_r, self.global_gap_scale)

    def runtime_metrics(self):
        metrics = super().runtime_metrics()
        total_scales = [
            self.global_gap_scale * scale for scale in self.gap_scales()
        ]
        metrics['correction'] = (
            sum(scale - 1.0 for scale in total_scales) / self.num_bins
        )
        return metrics

    def local_runtime_metrics(self):
        metrics = super().local_runtime_metrics()
        metrics['total_gap_scales'] = [
            self.global_gap_scale * scale for scale in metrics['gap_scales']
        ]
        return metrics

    def metadata(self):
        metadata = super().metadata()
        metadata.update(
            name=self.name,
            intervention='fixed_global_times_geometrically_neutral_local',
            global_gap_scale=self.global_gap_scale,
            **self.runtime_metrics(),
        )
        return metadata

def continuous_stage(cur_tick, double_ticks):
    """Legacy fractional-stage helper retained for import compatibility.

    adaptive_v1 now uses the official integer stage and adapts only from the
    loss EMA; new training code should not use this helper.
    """
    if double_ticks <= 0:
        raise ValueError(f'double_ticks must be > 0, got {double_ticks}')
    return cur_tick / double_ticks

#----------------------------------------------------------------------------
# Functional one-shot interface.

def compute_r(t, stage, schedule='sigmoid', **schedule_kwargs):
    """r = compute_r(t=t, stage=stage, schedule='sigmoid', q=256, k=8, b=1)"""
    return get_schedule(schedule, **schedule_kwargs).compute_r(t=t, stage=stage)

#----------------------------------------------------------------------------
# Quick visual check: python -m training.schedules

if __name__ == '__main__':
    t = torch.tensor([0.002, 0.02, 0.2, 2.0, 20.0, 80.0], dtype=torch.float64)
    print('r/t with q=2, k=8, b=1 at t =', t.tolist())
    for name in available_schedules():
        schedule = get_schedule(name)
        if name == 'adaptive_v1':
            schedule.update_training_signal(10.0)
            schedule.update_training_signal(7.0)
        print(f'--- {name} ---')
        for stage in [0, 1, 3, 7]:
            ratio = schedule.compute_r(t=t, stage=stage) / t
            print(f'  stage {stage:>4}: ' + '  '.join(f'{v:.4f}' for v in ratio.tolist()))
