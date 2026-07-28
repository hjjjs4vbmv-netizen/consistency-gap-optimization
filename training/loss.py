import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

from training.schedules import get_schedule

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0,
                 adj='sigmoid', adaptive_loss_ema_beta=0.9, adaptive_max_adjust=0.05,
                 adaptive_min_gap=1e-3, adaptive_warmup_updates=2,
                 local_tbin_num_bins=4, local_tbin_short_beta=0.9,
                 local_tbin_long_beta=0.99, local_tbin_warmup_updates=32,
                 local_tbin_gain=0.5, local_tbin_min_scale=0.75,
                 local_tbin_max_scale=1.5, local_tbin_deadband=0.02,
                 local_tbin_min_gap=1e-3, global_gap_scale=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        # t -> r entry point, dispatched through training/schedules.py.
        # 'const' / 'sigmoid' are the official fixed formulas (bit-identical
        # to the reference methods below); 'adaptive_v1' is the Role C
        # experiment.
        schedule_kwargs = dict(q=q, k=k, b=b)
        if adj == 'adaptive_v1':
            schedule_kwargs.update(
                loss_ema_beta=adaptive_loss_ema_beta,
                max_adjust=adaptive_max_adjust,
                min_gap=adaptive_min_gap,
                warmup_updates=adaptive_warmup_updates,
            )
        elif adj == 'global_sigmoid':
            schedule_kwargs.update(global_gap_scale=global_gap_scale)
        elif adj in ('local_tbin_v1', 'local_tbin_v2', 'local_tbin_v3'):
            schedule_kwargs.update(
                p_mean=P_mean,
                p_std=P_std,
                num_bins=local_tbin_num_bins,
                short_beta=local_tbin_short_beta,
                long_beta=local_tbin_long_beta,
                warmup_updates=local_tbin_warmup_updates,
                gain=local_tbin_gain,
                min_scale=local_tbin_min_scale,
                max_scale=local_tbin_max_scale,
                deadband=local_tbin_deadband,
                min_gap=local_tbin_min_gap,
            )
            if adj == 'local_tbin_v3':
                schedule_kwargs.update(global_gap_scale=global_gap_scale)
        self.schedule = get_schedule(adj, **schedule_kwargs)

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        self._runtime_r_over_t_mean = float('nan')
        self._runtime_gap_mean = float('nan')
        self._runtime_gap_over_sigmoid_gap_mean = float('nan')
        self._runtime_lower_gap_clip_rate = float('nan')
        self._runtime_upper_gap_clip_rate = float('nan')
        self._runtime_local_training_signal = None
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.schedule.update_schedule(stage)
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def update_training_signal(self, loss):
        return self.schedule.update_training_signal(loss)

    def schedule_state_dict(self):
        return {
            'schedule_name': self.schedule.name,
            'stage': self.stage,
            'ratio': self.ratio,
            'schedule': self.schedule.state_dict(),
        }

    def load_schedule_state_dict(self, state):
        saved_name = state.get('schedule_name')
        if saved_name is not None and saved_name != self.schedule.name:
            return False
        self.stage = state.get('stage', self.stage)
        self.ratio = state.get('ratio', self.ratio)
        self.schedule.load_state_dict(state.get('schedule', {}))
        return True

    def schedule_metadata(self):
        metadata = self.schedule.metadata()
        metadata.update(stage=self.stage, ratio=self.ratio)
        return metadata

    def schedule_runtime_metrics(self):
        """Return stable, scalar telemetry without exposing schedule internals."""
        metrics = self.schedule.runtime_metrics()
        return {
            'loss_ema': metrics['loss_ema'],
            'loss_reference': metrics['loss_reference'],
            'correction': float(metrics['correction']),
            'signal_updates': int(metrics['signal_updates']),
            'adaptive_active': bool(metrics['adaptive_active']),
            'r_over_t_mean': float(self._runtime_r_over_t_mean),
            'gap_mean': float(self._runtime_gap_mean),
            'gap_over_sigmoid_gap_mean': float(
                self._runtime_gap_over_sigmoid_gap_mean
            ),
            'lower_gap_clip_rate': float(
                self._runtime_lower_gap_clip_rate
            ),
            'upper_gap_clip_rate': float(
                self._runtime_upper_gap_clip_rate
            ),
        }

    def local_training_signal(self):
        """Return raw per-bin pair-loss sums/counts from the latest microbatch."""
        return self._runtime_local_training_signal

    def schedule_local_runtime_metrics(self):
        if hasattr(self.schedule, 'local_runtime_metrics'):
            return self.schedule.local_runtime_metrics()
        return None

    def _record_schedule_runtime_pair(self, t, r):
        with torch.no_grad():
            sigmoid_r = self.t_to_r_sigmoid(t)
            valid = (
                torch.isfinite(t)
                & torch.isfinite(r)
                & torch.isfinite(sigmoid_r)
                & (t > 0)
            )
            if not bool(valid.any()):
                self._runtime_r_over_t_mean = float('nan')
                self._runtime_gap_mean = float('nan')
                self._runtime_gap_over_sigmoid_gap_mean = float('nan')
                self._runtime_lower_gap_clip_rate = float('nan')
                self._runtime_upper_gap_clip_rate = float('nan')
                return
            valid_t = t[valid].to(torch.float64)
            valid_r = r[valid].to(torch.float64)
            valid_sigmoid_r = sigmoid_r[valid].to(torch.float64)
            realized_gap = (valid_t - valid_r).clamp_min(0)
            sigmoid_gap = (valid_t - valid_sigmoid_r).clamp_min(0)
            self._runtime_r_over_t_mean = float((valid_r / valid_t).mean().cpu())
            self._runtime_gap_mean = float((realized_gap / valid_t).mean().cpu())

            positive_sigmoid_gap = sigmoid_gap > 0
            if bool(positive_sigmoid_gap.any()):
                self._runtime_gap_over_sigmoid_gap_mean = float(
                    (
                        realized_gap[positive_sigmoid_gap]
                        / sigmoid_gap[positive_sigmoid_gap]
                    ).mean().cpu()
                )
            else:
                self._runtime_gap_over_sigmoid_gap_mean = float('nan')

            preclip_scale = self.schedule.preclip_gap_scale(t)
            if preclip_scale is None:
                self._runtime_lower_gap_clip_rate = float('nan')
                self._runtime_upper_gap_clip_rate = float('nan')
                return
            valid_scale = preclip_scale[valid].to(torch.float64)
            finite_scale = torch.isfinite(valid_scale) & positive_sigmoid_gap
            if not bool(finite_scale.any()):
                self._runtime_lower_gap_clip_rate = float('nan')
                self._runtime_upper_gap_clip_rate = float('nan')
                return
            intended_gap = sigmoid_gap[finite_scale] * valid_scale[finite_scale]
            compared_gap = realized_gap[finite_scale]
            compared_t = valid_t[finite_scale]
            source_dtype = (
                t.dtype if t.is_floating_point() else torch.get_default_dtype()
            )
            tolerance = (
                16
                * torch.finfo(source_dtype).eps
                * torch.maximum(
                    torch.maximum(intended_gap.abs(), compared_gap.abs()),
                    compared_t.abs(),
                )
            )
            self._runtime_lower_gap_clip_rate = float(
                (compared_gap > intended_gap + tolerance)
                .to(torch.float64)
                .mean()
                .cpu()
            )
            self._runtime_upper_gap_clip_rate = float(
                (compared_gap < intended_gap - tolerance)
                .to(torch.float64)
                .mean()
                .cpu()
            )

    # Official fixed t->r formulas, kept verbatim as the parity reference for
    # tests/test_schedules.py; the training path dispatches through
    # self.schedule (see __call__).
    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.schedule.compute_r(t=t, stage=self.stage)
        self._record_schedule_runtime_pair(t=t, r=r)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # Raw squared pair loss. Local t-bin schedules consume this signal before the
        # ECT sample-error transform and 1/(t-r) weighting are applied.
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        self._runtime_local_training_signal = None
        if self.schedule.name in ('local_tbin_v1', 'local_tbin_v2', 'local_tbin_v3'):
            with torch.no_grad():
                bin_ids = self.schedule.bin_indices(t).flatten()
                raw = loss.detach().to(torch.float64)
                sums = torch.zeros(self.schedule.num_bins, dtype=torch.float64, device=raw.device)
                counts = torch.zeros_like(sums)
                sums.scatter_add_(0, bin_ids, raw)
                counts.scatter_add_(0, bin_ids, torch.ones_like(raw))
                self._runtime_local_training_signal = {
                    'loss_sums': sums,
                    'loss_counts': counts,
                }
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()
