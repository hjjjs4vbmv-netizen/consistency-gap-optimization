import os
import csv
import time
import copy
import filecmp
import json
import math
import pickle
import random
import psutil
import shutil
import functools
import hashlib
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training import reproducibility

from metrics import metric_main

_STRICT_FACTORIAL_PROTOCOLS = {
    'q256_target_weight_v1',
    'q128_matched_spacing_v1',
}
_AUTHORITATIVE_TRANSFER_SOURCE_POLICY = {
    'schema': 'ect.q256.authoritative-transfer-source-policy/v1',
    'required_target_coverage': 'all_parameters_and_buffers',
    'allowed_source_extras': {
        'model.map_augment.weight': {
            'shape': [128, 9],
            'dtype': 'torch.float32',
            'tensor_bytes_sha256': (
                '4500f8ac1eb5cc8dd4096595a798c8ea4793d42f8433014ab67e41d5ceb70de0'
            ),
            'reason': 'authoritative checkpoint augmentation map unused by augment=0 target',
        },
    },
}

# Per-attempted-iteration CSV for paired fixed/adaptive comparisons.
# Schedule telemetry comes exclusively from loss_fn.schedule_runtime_metrics().
_LEGACY_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    'elapsed_sec',
    'peak_vram_gb',
)

# The telemetry schema predating next_loop_cur_tick. Keep this exact tuple so
# resumed runs can be migrated without guessing historical tick state.
_PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    'loss_ema',
    'loss_reference',
    'correction',
    'signal_updates',
    'adaptive_active',
    'r_over_t_mean',
    'gap_mean',
    'elapsed_sec',
    'peak_vram_gb',
)

# Schema used by the completed 2026-07 gap-factorial runs. Keep it exact so
# those checkpoints can resume after the clipping diagnostics were added.
_PRE_GAP_DIAGNOSTICS_TRAIN_SUMMARY_FIELDS = (
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'loss',
    'grad_scale',
    'step_skipped',
    'schedule',
    'stage',
    # The state that will be used by the next loop iteration. At a
    # maintenance boundary this is also the cur_tick persisted in a checkpoint.
    'next_loop_cur_tick',
    'loss_ema',
    'loss_reference',
    'correction',
    'signal_updates',
    'adaptive_active',
    'r_over_t_mean',
    'gap_mean',
    'elapsed_sec',
    'peak_vram_gb',
)

_TRAIN_SUMMARY_FIELDS = (
    *_PRE_GAP_DIAGNOSTICS_TRAIN_SUMMARY_FIELDS[:-2],
    'gap_over_sigmoid_gap_mean',
    'lower_gap_clip_rate',
    'upper_gap_clip_rate',
    *_PRE_GAP_DIAGNOSTICS_TRAIN_SUMMARY_FIELDS[-2:],
)

_FACTORIAL_TELEMETRY_FIELDS = (
    'schema',
    'protocol',
    'arm',
    'target_gap_scale',
    'denominator_gap_scale',
    'attempted_iteration',
    'successful_optimizer_steps',
    'processed_nimg',
    'processed_kimg',
    'stage',
    'loss',
    'loss_nonfinite_count',
    'raw_grad_norm',
    'raw_grad_finite_norm',
    'raw_grad_nonfinite_count',
    'sanitized_grad_norm',
    'sanitized_grad_nonfinite_count',
    'update_norm',
    'update_nonfinite_count',
    'model_norm',
    'model_nonfinite_count',
    'ema_norm',
    'ema_nonfinite_count',
    'sample_count',
    'batch_sha256',
    't_sha256',
    'base_r_sha256',
    'target_r_sha256',
    'denominator_r_sha256',
    'target_delta_sha256',
    'denominator_delta_sha256',
    'base_r_zero_count',
    'target_r_zero_count',
    'target_r_equal_t_count',
    'target_scaled_to_zero_count',
    'denominator_r_zero_count',
    'denominator_r_equal_t_count',
    'denominator_scaled_to_zero_count',
    'target_delta_min',
    'target_delta_max',
    'target_delta_mean',
    'denominator_delta_min',
    'denominator_delta_max',
    'denominator_delta_mean',
    'factor_nonfinite_count',
    'nonpositive_denominator_count',
    'learning_rate',
    'grad_scale_before',
    'grad_scale_after',
    'step_skipped',
    'elapsed_sec',
    'gpu_hours_cumulative',
)

#----------------------------------------------------------------------------

def canonical_processed_nimg(value):
    """Return the exact non-negative integer used by strict CSV contracts."""
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError('processed_nimg must not be boolean')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f'processed_nimg must be a finite non-negative integer: {value!r}'
        ) from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise RuntimeError(
            f'processed_nimg must be a finite non-negative integer: {value!r}'
        )
    return int(number)

#----------------------------------------------------------------------------

def load_and_migrate_train_summary(summary_path):
    """Load a resume CSV, upgrading only known historical schemas.

    Values absent from the original schema cannot be reconstructed, so their
    migrated cells deliberately stay empty. The original file is retained
    beside the upgraded CSV for auditability.
    """
    with open(summary_path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)

    if not rows:
        raise RuntimeError(f'resume requested but {summary_path} has no data rows')
    if fieldnames == _TRAIN_SUMMARY_FIELDS:
        return rows, None
    if fieldnames == _LEGACY_TRAIN_SUMMARY_FIELDS:
        backup_path = f'{summary_path}.pre-telemetry.bak'
    elif fieldnames == _PRE_NEXT_LOOP_TICK_TRAIN_SUMMARY_FIELDS:
        backup_path = f'{summary_path}.pre-next-loop-tick.bak'
    elif fieldnames == _PRE_GAP_DIAGNOSTICS_TRAIN_SUMMARY_FIELDS:
        backup_path = f'{summary_path}.pre-gap-diagnostics.bak'
    else:
        raise RuntimeError(
            f'resume requested but {summary_path} has an unsupported schema; '
            'expected the current schema or an exact supported legacy schema'
        )

    if os.path.exists(backup_path):
        if not filecmp.cmp(summary_path, backup_path, shallow=False):
            raise RuntimeError(
                f'refuse to overwrite non-matching train-summary backup: {backup_path}'
            )
    else:
        shutil.copy2(summary_path, backup_path)

    migrated_rows = [
        {field: row.get(field, '') for field in _TRAIN_SUMMARY_FIELDS}
        for row in rows
    ]
    temporary_path = f'{summary_path}.telemetry-migration.tmp-{os.getpid()}'
    try:
        with open(temporary_path, 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=_TRAIN_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(migrated_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, summary_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return migrated_rows, backup_path

#----------------------------------------------------------------------------

def adaptive_update_interval_nimg(update_kimg):
    """Convert an adaptive update period to an exact image-count interval."""
    update_kimg = float(update_kimg)
    update_nimg = update_kimg * 1000
    if not math.isfinite(update_kimg) or update_kimg <= 0 or not update_nimg.is_integer():
        raise ValueError(
            f'adaptive_update_kimg must be positive and represent whole images, got {update_kimg}'
        )
    return int(update_nimg)


class AdaptiveSignalWindow:
    """Accumulate local loss until the next absolute adaptive-update boundary.

    Windows are deliberately independent of maintenance ticks so changing
    --tick does not change the controller's update frequency.
    """

    def __init__(self, update_kimg, start_nimg=0):
        self.update_nimg = adaptive_update_interval_nimg(update_kimg)
        start_nimg = int(start_nimg)
        if start_nimg < 0:
            raise ValueError(f'start_nimg must be non-negative, got {start_nimg}')
        self.next_update_nimg = (start_nimg // self.update_nimg + 1) * self.update_nimg
        self.loss_sum = 0.0
        self.loss_count = 0

    def add(self, loss_sum, loss_count):
        self.loss_sum += float(loss_sum)
        self.loss_count += int(loss_count)

    def pop_if_due(self, cur_nimg):
        cur_nimg = int(cur_nimg)
        if cur_nimg < self.next_update_nimg:
            return None
        loss_sum, loss_count = self.loss_sum, self.loss_count
        self.loss_sum = 0.0
        self.loss_count = 0
        while self.next_update_nimg <= cur_nimg:
            self.next_update_nimg += self.update_nimg
        return loss_sum, loss_count

    def state_dict(self):
        """Return all state needed to resume a partially accumulated window."""
        return {
            'update_nimg': self.update_nimg,
            'next_update_nimg': self.next_update_nimg,
            'loss_sum': self.loss_sum,
            'loss_count': self.loss_count,
        }

    def load_state_dict(self, state):
        """Restore a window checkpointed after an arbitrary training step."""
        if not isinstance(state, dict):
            raise ValueError('adaptive signal window state must be a dict')
        required = ('update_nimg', 'next_update_nimg', 'loss_sum', 'loss_count')
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                f'adaptive signal window state missing required fields: {", ".join(missing)}'
            )
        update_nimg = int(state['update_nimg'])
        next_update_nimg = int(state['next_update_nimg'])
        loss_sum = float(state['loss_sum'])
        loss_count = int(state['loss_count'])
        if update_nimg != self.update_nimg:
            raise ValueError(
                f'adaptive signal window interval mismatch: checkpoint={update_nimg}, '
                f'current={self.update_nimg}'
            )
        if next_update_nimg <= 0 or next_update_nimg % self.update_nimg != 0:
            raise ValueError(
                f'invalid adaptive signal window next_update_nimg: {next_update_nimg}'
            )
        if loss_count < 0:
            raise ValueError(f'adaptive signal window loss_count must be non-negative, got {loss_count}')
        self.next_update_nimg = next_update_nimg
        self.loss_sum = loss_sum
        self.loss_count = loss_count


class LocalTBinSignalWindow:
    """Accumulate raw pair-loss sums/counts for equal-probability t bins."""

    def __init__(self, update_kimg, num_bins, start_nimg=0):
        self.update_nimg = adaptive_update_interval_nimg(update_kimg)
        if isinstance(num_bins, bool) or int(num_bins) != num_bins or int(num_bins) < 2:
            raise ValueError(f'num_bins must be an integer >= 2, got {num_bins}')
        self.num_bins = int(num_bins)
        start_nimg = int(start_nimg)
        if start_nimg < 0:
            raise ValueError(f'start_nimg must be non-negative, got {start_nimg}')
        self.next_update_nimg = (start_nimg // self.update_nimg + 1) * self.update_nimg
        self.loss_sums = [0.0] * self.num_bins
        self.loss_counts = [0] * self.num_bins

    def add(self, loss_sums, loss_counts):
        if len(loss_sums) != self.num_bins or len(loss_counts) != self.num_bins:
            raise ValueError('local t-bin signal size mismatch')
        for index in range(self.num_bins):
            self.loss_sums[index] += float(loss_sums[index])
            self.loss_counts[index] += int(loss_counts[index])

    def pop_if_due(self, cur_nimg):
        cur_nimg = int(cur_nimg)
        if cur_nimg < self.next_update_nimg:
            return None
        result = (list(self.loss_sums), list(self.loss_counts))
        self.loss_sums = [0.0] * self.num_bins
        self.loss_counts = [0] * self.num_bins
        while self.next_update_nimg <= cur_nimg:
            self.next_update_nimg += self.update_nimg
        return result

    def state_dict(self):
        return {
            'kind': 'local_tbin',
            'update_nimg': self.update_nimg,
            'next_update_nimg': self.next_update_nimg,
            'num_bins': self.num_bins,
            'loss_sums': list(self.loss_sums),
            'loss_counts': list(self.loss_counts),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError('local t-bin signal window state must be a dict')
        required = ('update_nimg', 'next_update_nimg', 'num_bins', 'loss_sums', 'loss_counts')
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                f'local t-bin signal window state missing required fields: {", ".join(missing)}'
            )
        update_nimg = int(state['update_nimg'])
        next_update_nimg = int(state['next_update_nimg'])
        num_bins = int(state['num_bins'])
        loss_sums = [float(value) for value in state['loss_sums']]
        loss_counts = [int(value) for value in state['loss_counts']]
        if update_nimg != self.update_nimg:
            raise ValueError(
                f'local t-bin window interval mismatch: checkpoint={update_nimg}, '
                f'current={self.update_nimg}'
            )
        if num_bins != self.num_bins or len(loss_sums) != num_bins or len(loss_counts) != num_bins:
            raise ValueError('local t-bin window bin count mismatch')
        if next_update_nimg <= 0 or next_update_nimg % self.update_nimg != 0:
            raise ValueError(
                f'invalid local t-bin window next_update_nimg: {next_update_nimg}'
            )
        if any(count < 0 for count in loss_counts):
            raise ValueError('local t-bin window counts must be non-negative')
        self.next_update_nimg = next_update_nimg
        self.loss_sums = loss_sums
        self.loss_counts = loss_counts


def gather_adaptive_signal_window_state(window, device):
    """Collect each rank's local adaptive-window state for a rank-0 checkpoint."""
    local_state = window.state_dict()
    world_size = dist.get_world_size()
    if world_size == 1:
        rank_states = [local_state]
    elif isinstance(window, LocalTBinSignalWindow):
        local_values = torch.tensor(
            [window.next_update_nimg, *window.loss_sums, *window.loss_counts],
            dtype=torch.float64,
            device=device,
        )
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_values, local_values)
        rank_states = []
        for values in gathered_values:
            sums_start = 1
            counts_start = sums_start + window.num_bins
            rank_states.append({
                'kind': 'local_tbin',
                'update_nimg': window.update_nimg,
                'next_update_nimg': int(values[0]),
                'num_bins': window.num_bins,
                'loss_sums': [float(value) for value in values[sums_start:counts_start]],
                'loss_counts': [int(value) for value in values[counts_start:]],
            })
    else:
        local_values = torch.tensor(
            [window.next_update_nimg, window.loss_sum, window.loss_count],
            dtype=torch.float64,
            device=device,
        )
        gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_values, local_values)
        rank_states = [
            {
                'update_nimg': window.update_nimg,
                'next_update_nimg': int(values[0]),
                'loss_sum': float(values[1]),
                'loss_count': int(values[2]),
            }
            for values in gathered_values
        ]

    # Keep the rank-0 fields at the top level for transparent single-rank
    # inspection, and retain every local accumulator for exact DDP resumes.
    return {**rank_states[0], 'rank_states': rank_states}


def local_adaptive_signal_window_state(state):
    """Select this rank's window state from a training-state checkpoint."""
    if not isinstance(state, dict):
        return state
    rank_states = state.get('rank_states')
    if rank_states is None:
        return state
    if not isinstance(rank_states, list) or len(rank_states) != dist.get_world_size():
        raise ValueError(
            'adaptive signal window checkpoint rank count does not match the current world size'
        )
    return rank_states[dist.get_rank()]


def globally_average_adaptive_loss(loss_sum, loss_count, device):
    """Return the sample-weighted loss mean, identical on every rank."""
    totals = torch.tensor([loss_sum, loss_count], dtype=torch.float64, device=device)
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(totals)
    total_count = float(totals[1])
    return float(totals[0] / total_count) if total_count > 0 else float('nan')


def globally_average_local_tbin_loss(loss_sums, loss_counts, device):
    """Return per-bin raw-loss means, sample weighted across DDP ranks."""
    if len(loss_sums) != len(loss_counts):
        raise ValueError('local t-bin sums/counts size mismatch')
    totals = torch.tensor(
        [*loss_sums, *loss_counts], dtype=torch.float64, device=device
    )
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(totals)
    num_bins = len(loss_sums)
    means = []
    for index in range(num_bins):
        count = float(totals[num_bins + index])
        means.append(float(totals[index] / count) if count > 0 else None)
    return means


def globally_average_runtime_pairs(metric_batches, device):
    """Average public realized-pair telemetry across rounds and ranks."""
    fields = (
        'r_over_t_mean',
        'gap_mean',
        'gap_over_sigmoid_gap_mean',
        'lower_gap_clip_rate',
        'upper_gap_clip_rate',
    )
    sums_and_counts = []
    for field in fields:
        values = [float(metrics[field]) for metrics in metric_batches]
        values = [value for value in values if math.isfinite(value)]
        sums_and_counts.extend((sum(values), len(values)))
    totals = torch.tensor(
        sums_and_counts,
        dtype=torch.float64,
        device=device,
    )
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(totals)
    result = {}
    for index, field in enumerate(fields):
        total = totals[index * 2]
        count = float(totals[index * 2 + 1])
        result[field] = (
            float(total / count) if count > 0 else float('nan')
        )
    return result


def gather_rank_reproducibility_state(
    dataset_sampler, consumed_samples, *, device=None
):
    """Collect ordered, logical per-rank state before any preview/evaluation RNG."""
    rng_state = (
        reproducibility.capture_rng_state()
        if device is None
        else reproducibility.capture_current_device_rng_state(device)
    )
    local_state = {
        'rank': dist.get_rank(),
        'world_size': dist.get_world_size(),
        'rng_state': rng_state,
        'sampler_state': dataset_sampler.state_dict(
            consumed_samples=consumed_samples
        ),
    }
    if dist.get_world_size() == 1:
        states = [local_state]
    else:
        states = [None] * dist.get_world_size()
        torch.distributed.all_gather_object(states, local_state)
    if len(states) != dist.get_world_size():
        raise RuntimeError('reproducibility rank-state count mismatch')
    for expected_rank, state in enumerate(states):
        if not isinstance(state, dict):
            raise RuntimeError('reproducibility rank state must be a dict')
        if int(state.get('rank', -1)) != expected_rank:
            raise RuntimeError(
                'reproducibility rank states are not in canonical rank order'
            )
        if int(state.get('world_size', -1)) != dist.get_world_size():
            raise RuntimeError('reproducibility world-size mismatch')
    return states


def select_local_reproducibility_state(states):
    if not isinstance(states, list) or len(states) != dist.get_world_size():
        raise RuntimeError(
            'training-state rank count does not match current world size'
        )
    for expected_rank, state in enumerate(states):
        if not isinstance(state, dict) or int(state.get('rank', -1)) != expected_rank:
            raise RuntimeError('training-state rank ordering is invalid')
        if int(state.get('world_size', -1)) != dist.get_world_size():
            raise RuntimeError('training-state rank world size is invalid')
    return states[dist.get_rank()]


@torch.no_grad()
def copy_module_state_exact(
    src_module, dst_module, *, label, allowed_source_extras=None,
    allow_unlisted_source_extras=False,
):
    """Copy every destination tensor after fail-closed source validation."""
    if not isinstance(src_module, torch.nn.Module):
        raise RuntimeError(f'{label} source is not a torch module')
    if not isinstance(dst_module, torch.nn.Module):
        raise RuntimeError(f'{label} destination is not a torch module')
    src_items = list(misc.named_params_and_buffers(src_module))
    dst_items = list(misc.named_params_and_buffers(dst_module))
    src = dict(src_items)
    dst = dict(dst_items)
    if len(src) != len(src_items) or len(dst) != len(dst_items):
        raise RuntimeError(f'{label} has duplicate parameter/buffer names')
    missing = sorted(set(dst) - set(src))
    extra = sorted(set(src) - set(dst))
    allowed_source_extras = (
        {} if allowed_source_extras is None else dict(allowed_source_extras)
    )
    extras_mismatch = (
        not allow_unlisted_source_extras
        and set(extra) != set(allowed_source_extras)
    )
    if missing or extras_mismatch:
        raise RuntimeError(
            f'{label} parameter/buffer key mismatch: '
            f'missing={missing}, extra={extra}, '
            f'allowed_source_extras={sorted(allowed_source_extras)}'
        )
    for name in (() if allow_unlisted_source_extras else extra):
        record = allowed_source_extras[name]
        if not isinstance(record, dict) or set(record) != {
            'shape', 'dtype', 'tensor_bytes_sha256', 'reason'
        }:
            raise RuntimeError(
                f'{label} invalid source-extra policy for {name}'
            )
        source = src[name].detach().cpu().contiguous()
        actual = {
            'shape': list(source.shape),
            'dtype': str(source.dtype),
            'tensor_bytes_sha256': hashlib.sha256(
                source.numpy().tobytes()
            ).hexdigest(),
            'reason': record['reason'],
        }
        if actual != record:
            raise RuntimeError(
                f'{label} source-extra identity mismatch for {name}: '
                f'{actual} != {record}'
            )
    for name in sorted(dst):
        source = src[name]
        target = dst[name]
        if source.shape != target.shape:
            raise RuntimeError(
                f'{label} tensor shape mismatch for {name}: '
                f'{tuple(source.shape)} != {tuple(target.shape)}'
            )
        if source.dtype != target.dtype:
            raise RuntimeError(
                f'{label} tensor dtype mismatch for {name}: '
                f'{source.dtype} != {target.dtype}'
            )
        target.copy_(source.detach())


@torch.no_grad()
def tensor_collection_diagnostics(tensors):
    """Return total non-finite count, mathematical norm, and finite-part norm."""
    iterator = iter(tensors)
    first = next(iterator, None)
    if first is None:
        return 0, 0.0, 0.0
    device = first.device
    nonfinite_count = torch.zeros([], dtype=torch.int64, device=device)
    finite_square_sum = torch.zeros([], dtype=torch.float64, device=device)

    def accumulate(tensor):
        nonlocal nonfinite_count, finite_square_sum
        value = tensor.detach()
        finite = torch.isfinite(value)
        nonfinite_count += (~finite).sum()
        finite_value = torch.where(finite, value, torch.zeros_like(value))
        finite_square_sum += finite_value.to(torch.float64).square().sum()

    accumulate(first)
    for tensor in iterator:
        accumulate(tensor)
    packed = torch.stack(
        [nonfinite_count.to(torch.float64), finite_square_sum]
    ).cpu()
    count = int(packed[0])
    finite_norm = math.sqrt(float(packed[1]))
    norm = float('inf') if count else finite_norm
    return count, norm, finite_norm


@torch.no_grad()
def tensor_collection_nonfinite_count(tensors):
    """Count non-finite values without the strict protocol's norm telemetry."""
    iterator = iter(tensors)
    first = next(iterator, None)
    if first is None:
        return 0
    count = torch.zeros([], dtype=torch.int64, device=first.device)
    count += (~torch.isfinite(first.detach())).sum()
    for tensor in iterator:
        count += (~torch.isfinite(tensor.detach())).sum()
    return int(count.cpu())


def globally_sum_counts(values, device):
    counts = torch.tensor(tuple(int(value) for value in values), dtype=torch.int64, device=device)
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(counts)
    return tuple(int(value) for value in counts.cpu())


def enforce_generic_exact_finite(stage, diagnostics):
    failures = [name for name, count in diagnostics.items() if int(count)]
    if failures:
        raise FloatingPointError(
            f'generic exact non-finite {stage}: {", ".join(failures)}'
        )


def enforce_generic_exact_finite_before_sanitization(losses, parameters, device):
    loss_count = tensor_collection_nonfinite_count(losses)
    gradient_count = tensor_collection_nonfinite_count(
        param.grad for param in parameters if param.grad is not None
    )
    loss_count, gradient_count = globally_sum_counts(
        (loss_count, gradient_count), device
    )
    enforce_generic_exact_finite(
        'loss/gradient before sanitization',
        {'loss': loss_count, 'raw gradient': gradient_count},
    )
    return loss_count, gradient_count


def aggregate_factorial_runtime_metrics(metric_batches):
    if not metric_batches or any(item is None for item in metric_batches):
        raise RuntimeError('strict factorial loss did not emit runtime telemetry')
    identity_fields = (
        'schema', 'protocol', 'arm', 'target_gap_scale',
        'denominator_gap_scale',
    )
    first = metric_batches[0]
    for metrics in metric_batches[1:]:
        for field in identity_fields:
            if metrics[field] != first[field]:
                raise RuntimeError(
                    f'factorial runtime telemetry changed {field} within an attempt'
                )
    count_fields = (
        'sample_count', 'base_r_zero_count', 'target_r_zero_count',
        'target_r_equal_t_count', 'target_scaled_to_zero_count',
        'denominator_r_zero_count', 'denominator_r_equal_t_count',
        'denominator_scaled_to_zero_count', 'nonfinite_count',
        'nonpositive_denominator_count',
    )
    result = {field: first[field] for field in identity_fields}
    for field in count_fields:
        result[field] = sum(int(metrics[field]) for metrics in metric_batches)
    for field in (
        't_sha256', 'base_r_sha256', 'target_r_sha256',
        'denominator_r_sha256', 'target_delta_sha256',
        'denominator_delta_sha256',
    ):
        result[field] = reproducibility.state_sha256(
            [metrics[field] for metrics in metric_batches]
        )
    for prefix in ('target_delta', 'denominator_delta'):
        result[f'{prefix}_min'] = min(
            float(metrics[f'{prefix}_min']) for metrics in metric_batches
        )
        result[f'{prefix}_max'] = max(
            float(metrics[f'{prefix}_max']) for metrics in metric_batches
        )
        total_count = result['sample_count']
        result[f'{prefix}_mean'] = sum(
            float(metrics[f'{prefix}_mean']) * int(metrics['sample_count'])
            for metrics in metric_batches
        ) / total_count
    return result


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def normalize_immutable_checkpoint_nimg(
    milestone_kimg, *, total_kimg, batch_size
):
    """Validate I/O-only milestones and return their exact image counts."""
    values = tuple(milestone_kimg or ())
    if len(set(values)) != len(values):
        raise ValueError('immutable checkpoint kimg values must be unique')
    if values != tuple(sorted(values)):
        raise ValueError('immutable checkpoint kimg values must be increasing')
    result = []
    for value in values:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(
                'immutable checkpoint kimg values must be positive integers'
            )
        nimg = int(value) * 1000
        if nimg > int(total_kimg) * 1000:
            raise ValueError(
                f'immutable checkpoint {value} kimg exceeds total budget '
                f'{total_kimg} kimg'
            )
        if nimg % int(batch_size) != 0:
            raise ValueError(
                f'immutable checkpoint {value} kimg is not reachable with '
                f'batch_size={batch_size}'
            )
        result.append(nimg)
    return tuple(result)


def resolve_batch_layout(batch_size, batch_gpu, world_size):
    """Return per-rank microbatch layout without silently dropping samples."""
    if world_size <= 0 or batch_size <= 0:
        raise ValueError('batch_size and world_size must be positive')
    if batch_size % world_size != 0:
        raise ValueError(
            f'batch_size={batch_size} is not divisible by world_size={world_size}'
        )
    per_rank = batch_size // world_size
    if batch_gpu is None or batch_gpu > per_rank:
        batch_gpu = per_rank
    if batch_gpu <= 0 or per_rank % batch_gpu != 0:
        raise ValueError(
            f'per-rank batch {per_rank} is not divisible by batch_gpu={batch_gpu}'
        )
    return batch_gpu, per_rank // batch_gpu


def learning_rate_schedule(
    cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3,
    rampup_kimg=None,
):
    """Inverse-square-root schedule used by the EDM2 ImageNet recipe."""
    learning_rate = float(ref_lr)
    if ref_batches > 0:
        learning_rate /= math.sqrt(
            max(cur_nimg / (ref_batches * batch_size), 1)
        )
    if rampup_kimg:
        learning_rate *= min(cur_nimg / (rampup_kimg * 1000), 1)
    return learning_rate


def immutable_training_state_path(run_dir, cur_nimg):
    if int(cur_nimg) != cur_nimg or int(cur_nimg) <= 0:
        raise ValueError('immutable checkpoint image count must be positive')
    cur_nimg = int(cur_nimg)
    if cur_nimg % 1000 != 0:
        raise ValueError('immutable checkpoint image count must be whole kimg')
    return os.path.join(
        run_dir, f'training-state-kimg{cur_nimg // 1000:06d}.pt'
    )


def save_immutable_training_state(state, run_dir, cur_nimg):
    path = immutable_training_state_path(run_dir, cur_nimg)
    reproducibility.atomic_torch_save(state, path, overwrite=False)
    return path

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, 
    t_max=80, mid_t=None
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    lr_kwargs           = None,     # Optional inverse-square-root LR schedule.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_beta            = 0.9999,   # EMA decay rate. Overwritten by ema_halflife_kimg.
    ema_halflife_kimg   = None,     # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = None,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 0,        # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 500,      # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    ckpt_ticks          = 100,      # How often to save latest checkpoints, None = disable.
    immutable_checkpoint_kimg = (), # Exact I/O-only full-state milestones.
    sample_ticks        = 50,       # How often to sample images, None = disable.
    eval_ticks          = 500,      # How often to evaluate models, None = disable.
    double_ticks        = 500,      # How often to evaluate models, None = disable.
    adaptive_update_kimg = 0.5,     # Adaptive loss-EMA signal period, independent of ticks.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_tick         = 0,        # Start from the given training progress.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    enable_tf32         = False,    # Enable tf32 for A100/H100 GPUs?
    enable_amp          = False,    # Enable torch.cuda.amp.GradScaler
    exact_resume        = False,    # Topology-bound full-state exact replay.
    global_batch_mean   = False,    # Normalize gradients across microbatches.
    power_ema_stds      = (),       # Optional PowerEMA relative std profiles.
    startup_preview     = True,     # Write initial data/model image grids.
    stop_after_attempts = None,     # Gate-only planned pause after N attempts.
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    strict_reproducibility = (
        loss_kwargs.get('factorial_protocol') in _STRICT_FACTORIAL_PROTOCOLS
    )
    generic_exact_resume = bool(exact_resume and not strict_reproducibility)
    exact_reproducibility = strict_reproducibility or generic_exact_resume
    if strict_reproducibility and dist.get_world_size() != 1:
        raise ValueError(
            'formal q256 target-weight arms require one process and one '
            'exclusive GPU per run'
        )
    if strict_reproducibility and not enable_amp:
        raise ValueError(
            'formal q256 target-weight arms require AMP/GradScaler enabled'
        )
    if generic_exact_resume and not global_batch_mean:
        raise ValueError(
            'generic exact resume requires global_batch_mean=True'
        )
    if stop_after_attempts is not None:
        if not exact_reproducibility:
            raise ValueError(
                'stop_after_attempts is reserved for exact resume gates'
            )
        if (
            isinstance(stop_after_attempts, bool)
            or int(stop_after_attempts) != stop_after_attempts
            or int(stop_after_attempts) != 16
        ):
            raise ValueError(
                'stop_after_attempts is frozen to 16 for the exact 16+16 gate'
            )
        stop_after_attempts = int(stop_after_attempts)
    immutable_checkpoint_nimg = normalize_immutable_checkpoint_nimg(
        immutable_checkpoint_kimg,
        total_kimg=total_kimg,
        batch_size=batch_size,
    )
    if immutable_checkpoint_nimg and not exact_reproducibility:
        raise ValueError(
            'immutable checkpoint milestones require exact resume'
        )
    rank_seed = (seed * dist.get_world_size() + dist.get_rank()) % (1 << 31)
    np.random.seed(rank_seed)
    if exact_reproducibility:
        random.seed(rank_seed)
    torch.manual_seed(np.random.randint(1 << 31))
    if exact_reproducibility:
        if cudnn_benchmark:
            message = (
                'formal q256 target-weight arms'
                if strict_reproducibility else 'generic exact resume'
            )
            raise ValueError(f'{message} require cudnn_benchmark=False for exact replay')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            message = (
                'formal q256 target-weight arms'
                if strict_reproducibility else 'generic exact resume'
            )
            raise ValueError(
                f'{message} require CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                'for exact replay'
            )
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Enable these to speed up on A100 GPUs
    dist.print0(f'Enable tf32: {enable_tf32}')
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enable_tf32

    # Select batch size per GPU.
    batch_gpu, num_accumulation_rounds = resolve_batch_layout(
        batch_size, batch_gpu, dist.get_world_size()
    )
    strict_trajectory_config = None
    strict_trajectory_config_sha256 = None
    if exact_reproducibility:
        trajectory_data = {
            'schema': (
                reproducibility.TRAJECTORY_CONFIG_SCHEMA
                if strict_reproducibility
                else reproducibility.EXACT_TRAJECTORY_CONFIG_SCHEMA
            ),
            'seed': seed,
            'rank_seed': rank_seed,
            'world_size': dist.get_world_size(),
            'batch_size': batch_size,
            'batch_gpu': batch_gpu,
            'num_accumulation_rounds': num_accumulation_rounds,
            'total_kimg': total_kimg,
            'ema_beta': ema_beta,
            'ema_halflife_kimg': ema_halflife_kimg,
            'ema_rampup_ratio': ema_rampup_ratio,
            'lr_rampup_kimg': lr_rampup_kimg,
            'loss_scaling': loss_scaling,
            'kimg_per_tick': kimg_per_tick,
            'snapshot_ticks': snapshot_ticks,
            'state_dump_ticks': state_dump_ticks,
            'ckpt_ticks': ckpt_ticks,
            'sample_ticks': sample_ticks,
            'eval_ticks': eval_ticks,
            'double_ticks': double_ticks,
            'adaptive_update_kimg': adaptive_update_kimg,
            'mid_t': mid_t,
            'metrics': metrics,
            'cudnn_benchmark': cudnn_benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': (
                torch.are_deterministic_algorithms_enabled()
            ),
            'cublas_workspace_config': os.environ.get(
                'CUBLAS_WORKSPACE_CONFIG'
            ),
            'enable_tf32': enable_tf32,
            'enable_amp': enable_amp,
            'device': str(device),
            'dataset_kwargs': dict(dataset_kwargs),
            'data_loader_kwargs': dict(data_loader_kwargs),
            'network_kwargs': dict(network_kwargs),
            'loss_kwargs': dict(loss_kwargs),
            'optimizer_kwargs': dict(optimizer_kwargs),
            'augment_kwargs': (
                None if augment_kwargs is None else dict(augment_kwargs)
            ),
            'authoritative_transfer_source_policy': copy.deepcopy(
                _AUTHORITATIVE_TRANSFER_SOURCE_POLICY
            ),
        }
        if generic_exact_resume:
            trajectory_data.pop('rank_seed')
            trajectory_data.update(
                exact_resume=True,
                rank_seeds=[
                    (seed * dist.get_world_size() + rank) % (1 << 31)
                    for rank in range(dist.get_world_size())
                ],
                global_batch_mean=global_batch_mean,
                lr_kwargs=(None if lr_kwargs is None else dict(lr_kwargs)),
                power_ema_stds=tuple(power_ema_stds),
                immutable_checkpoint_kimg=tuple(immutable_checkpoint_kimg),
                startup_preview=startup_preview,
            )
        strict_trajectory_config = reproducibility.canonical_json_data(
            trajectory_data
        )
        strict_trajectory_config_sha256 = reproducibility.state_sha256(
            strict_trajectory_config
        )

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    # A strict resume must load the logical sampler cursor before creating the
    # iterator; otherwise DataLoader prefetch would enqueue examples from zero.
    dataset_iterator = None
    if not (exact_reproducibility and resume_state_dump):
        dataset_iterator = iter(torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        ))
    local_consumed_samples = 0

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    # Automatic Mixed Precision
    dist.print0(f'GradScaler enabled: {enable_amp} for mixed precision training')
    if enable_amp:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        dist.print0('Setting up GradScaler...')
        scaler = torch.cuda.amp.GradScaler()
        dist.print0('Loss scaling is overwritten when GradScaler is enabled')

    dist.print0('Setting up DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    
    # Stats
    if dist.get_rank() == 0 and not (
        exact_reproducibility and resume_state_dump
    ):
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        if strict_reproducibility:
            copy_module_state_exact(
                data.get('ema'), net, label='authoritative transfer -> net',
                allowed_source_extras=(
                    _AUTHORITATIVE_TRANSFER_SOURCE_POLICY[
                        'allowed_source_extras'
                    ]
                ),
            )
            copy_module_state_exact(
                data.get('ema'), ema, label='authoritative transfer -> EMA',
                allowed_source_extras=(
                    _AUTHORITATIVE_TRANSFER_SOURCE_POLICY[
                        'allowed_source_extras'
                    ]
                ),
            )
        elif network_kwargs.get('class_name') == 'training.networks_edm2.Precond':
            copy_module_state_exact(
                data.get('ema'), net, label='EDM2 donor transfer -> net',
                allow_unlisted_source_extras=True,
            )
            copy_module_state_exact(
                data.get('ema'), ema, label='EDM2 donor transfer -> EMA',
                allow_unlisted_source_extras=True,
            )
        else:
            misc.copy_params_and_buffers(
                src_module=data['ema'], dst_module=net, require_all=False
            )
            misc.copy_params_and_buffers(
                src_module=data['ema'], dst_module=ema, require_all=False
            )
        del data # conserve memory
    power_ema = None
    if power_ema_stds:
        power_ema = dnnlib.util.construct_class_by_name(
            class_name='training.phema.PowerFunctionEMA',
            net=net,
            stds=tuple(power_ema_stds),
        )
        power_ema.reset()
    attempted_iteration = 0
    successful_optimizer_steps = 0
    resumed_cur_nimg = None
    resumed_cur_tick = None
    resumed_tick_start_nimg = None
    resumed_adaptive_signal_window_state = None
    resumed_rng_state = None
    resumed_snapshot_grid_z = None
    resumed_snapshot_grid_c = None
    resumed_snapshot_grid_size = None
    elapsed_base_sec = 0.0
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        # The training-state contains optimizer and persistent module objects.
        # Only load trusted checkpoints produced by this repository.
        data = torch.load(
            resume_state_dump,
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        if exact_reproducibility:
            expected_schema = (
                reproducibility.TRAINING_STATE_SCHEMA
                if strict_reproducibility
                else reproducibility.EXACT_TRAINING_STATE_SCHEMA
            )
            if data.get('reproducibility_schema') != expected_schema:
                raise RuntimeError(
                    'exact resume requires a complete versioned '
                    'training-state; legacy state is not replayable'
                )
            required = [
                'net', 'ema', 'optimizer_state', 'loss_fn_state',
                'rank_states',
                'attempted_iteration', 'successful_optimizer_steps',
                'cur_nimg', 'cur_tick', 'tick_start_nimg',
                'snapshot_grid_z', 'snapshot_grid_c', 'snapshot_grid_size',
                'trajectory_config', 'trajectory_config_sha256',
            ]
            if strict_reproducibility:
                required.extend(('factorial', 'gradscaler_state'))
            if power_ema is not None:
                required.append('power_ema_state')
            missing = [name for name in required if name not in data]
            if missing:
                raise RuntimeError(
                    'exact training-state missing fields: '
                    + ', '.join(missing)
                )
            if strict_reproducibility and data['factorial'] != loss_fn.factorial:
                raise RuntimeError(
                    'factorial factors in training-state do not match current config'
                )
            saved_trajectory_sha256 = reproducibility.state_sha256(
                data['trajectory_config']
            )
            if saved_trajectory_sha256 != data['trajectory_config_sha256']:
                raise RuntimeError(
                    'training-state trajectory config hash is internally invalid'
                )
            if saved_trajectory_sha256 != strict_trajectory_config_sha256:
                saved_trajectory_config = reproducibility.canonical_json_data(
                    data['trajectory_config']
                )
                current_trajectory_config = reproducibility.canonical_json_data(
                    strict_trajectory_config
                )
                saved_total_kimg = saved_trajectory_config.pop(
                    'total_kimg', None
                )
                current_total_kimg = current_trajectory_config.pop(
                    'total_kimg', None
                )
                completed_saved_budget = (
                    isinstance(saved_total_kimg, int)
                    and int(data['cur_nimg']) == saved_total_kimg * 1000
                )
                valid_budget_extension = (
                    saved_trajectory_config == current_trajectory_config
                    and isinstance(current_total_kimg, int)
                    and isinstance(saved_total_kimg, int)
                    and current_total_kimg > saved_total_kimg
                    and completed_saved_budget
                )
                if not valid_budget_extension:
                    raise RuntimeError(
                        'training-state trajectory config does not match current run'
                    )
                dist.print0(
                    'Extending completed exact training budget from '
                    f'{saved_total_kimg} to {current_total_kimg} kimg; '
                    'all other trajectory settings match exactly.'
                )
        if exact_reproducibility:
            copy_module_state_exact(
                data.get('net'), net, label='exact training-state -> net'
            )
        else:
            misc.copy_params_and_buffers(
                src_module=data['net'], dst_module=net, require_all=True
            )
        if exact_reproducibility:
            copy_module_state_exact(
                data.get('ema'), ema, label='exact training-state -> EMA'
            )
        if power_ema is not None and 'power_ema_state' in data:
            power_ema.load_state_dict(data['power_ema_state'])
        optimizer.load_state_dict(data['optimizer_state'])
        if 'cur_nimg' not in data:
            raise RuntimeError(
                f'resume training-state missing cur_nimg: {resume_state_dump}; '
                f'refuse filename-derived progress fallback for paired runs'
            )
        attempted_iteration = int(data.get('attempted_iteration', 0))
        successful_optimizer_steps = int(data.get('successful_optimizer_steps', 0))
        resumed_cur_nimg = int(data['cur_nimg'])
        if exact_reproducibility:
            local_rank_state = select_local_reproducibility_state(
                data['rank_states']
            )
            dataset_sampler.load_state_dict(local_rank_state['sampler_state'])
            local_consumed_samples = int(
                local_rank_state['sampler_state']['consumed_samples']
            )
            if resumed_cur_nimg % dist.get_world_size() != 0:
                raise RuntimeError(
                    'exact cur_nimg is not divisible by world size'
                )
            expected_consumed = resumed_cur_nimg // dist.get_world_size()
            if local_consumed_samples != expected_consumed:
                raise RuntimeError(
                    'sampler consumed_samples does not match restored cur_nimg: '
                    f'{local_consumed_samples} != {expected_consumed}'
                )
            resumed_rng_state = local_rank_state['rng_state']
            resumed_snapshot_grid_z = data['snapshot_grid_z']
            resumed_snapshot_grid_c = data['snapshot_grid_c']
            resumed_snapshot_grid_size = tuple(data['snapshot_grid_size'])
        if 'cur_tick' in data:
            resumed_cur_tick = int(data['cur_tick'])
        if 'tick_start_nimg' in data:
            resumed_tick_start_nimg = int(data['tick_start_nimg'])
        elapsed_base_sec = float(data.get('elapsed_sec', 0.0))
        if hasattr(loss_fn, 'load_schedule_state_dict') and 'loss_fn_state' in data:
            loaded = loss_fn.load_schedule_state_dict(data['loss_fn_state'])
            if exact_reproducibility and loaded is not True:
                raise RuntimeError(
                    'exact loss schedule state is incompatible'
                )
        if 'adaptive_signal_window_state' in data:
            resumed_adaptive_signal_window_state = data['adaptive_signal_window_state']
        if enable_amp:
            if 'gradscaler_state' in data:
                # NOTE(aiihn): Although not loading the state_dict of the GradScaler works well,
                # loading it can improve reproducibility.
                dist.print0(f'Loading GradScaler state from "{resume_state_dump}"...')
                scaler.load_state_dict(data['gradscaler_state'])
            else:
                if exact_reproducibility:
                    raise RuntimeError(
                        'exact training-state is missing GradScaler state'
                    )
                dist.print0(f'GradScaler state is not found in "{resume_state_dump}", using the default state.')
        del data # conserve memory

    if dataset_iterator is None:
        if not (exact_reproducibility and resume_state_dump):
            raise RuntimeError('dataset iterator was not initialized')
        dataset_iterator = iter(torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        ))
    
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        write_startup_preview = not (
            exact_reproducibility and resume_state_dump
        ) and startup_preview
        if write_startup_preview:
            dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        if resumed_snapshot_grid_size is not None:
            if tuple(grid_size) != resumed_snapshot_grid_size:
                raise RuntimeError('snapshot grid size changed across resume')
        if write_startup_preview:
            save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        if resumed_snapshot_grid_z is None:
            grid_z = torch.randn([labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
            grid_z = grid_z.split(batch_gpu)
            grid_c = torch.from_numpy(labels).to(device)
            grid_c = grid_c.split(batch_gpu)
        else:
            if resumed_snapshot_grid_c is None:
                raise RuntimeError('resumed snapshot grid labels are missing')
            grid_z = tuple(value.to(device) for value in resumed_snapshot_grid_z)
            grid_c = tuple(value.to(device) for value in resumed_snapshot_grid_c)
        
        if write_startup_preview:
            images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, 'model_init.png'), drange=[-1,1], grid_size=grid_size)
            del images

    # DataLoader worker seeding, module-summary dropout, and startup previews
    # are intentionally outside the resumed trajectory. Restore only after
    # all of them complete and immediately before training setup/iteration.
    if resumed_rng_state is not None:
        if dist.get_world_size() > 1:
            torch.distributed.barrier()
        if generic_exact_resume:
            reproducibility.restore_current_device_rng_state(
                resumed_rng_state, device
            )
        else:
            reproducibility.restore_rng_state(resumed_rng_state)
        if dist.get_world_size() > 1:
            torch.distributed.barrier()

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    # Prefer exact progress from training-state; filename-derived resume_tick is only a fallback.
    if resumed_cur_nimg is not None:
        cur_nimg = canonical_processed_nimg(resumed_cur_nimg)
    else:
        cur_nimg = canonical_processed_nimg(
            resume_tick * kimg_per_tick * 1000
        )
    if resumed_cur_tick is not None:
        cur_tick = resumed_cur_tick
    else:
        cur_tick = resume_tick
    if resumed_tick_start_nimg is not None:
        tick_start_nimg = resumed_tick_start_nimg
    else:
        tick_start_nimg = cur_nimg
    for milestone_nimg in immutable_checkpoint_nimg:
        milestone_path = immutable_training_state_path(
            run_dir, milestone_nimg
        )
        if milestone_nimg <= cur_nimg and not os.path.isfile(milestone_path):
            raise RuntimeError(
                'restored progress has passed immutable checkpoint without '
                f'its artifact: {milestone_path}'
            )
        if milestone_nimg > cur_nimg and os.path.exists(milestone_path):
            raise RuntimeError(
                'future immutable checkpoint already exists before replay '
                f'reaches it: {milestone_path}'
            )
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_kimg)
    stats_jsonl = None
    train_summary_csv = None
    train_summary_writer = None
    schedule_name = getattr(getattr(loss_fn, 'schedule', None), 'name', None)
    if schedule_name is None:
        schedule_name = getattr(loss_fn, 'adj', None)
    if schedule_name is None:
        schedule_name = loss_kwargs.get('adj', 'unknown')
    if schedule_name == 'adaptive_v1':
        adaptive_signal_window = AdaptiveSignalWindow(
            adaptive_update_kimg, start_nimg=cur_nimg
        )
    elif schedule_name in ('local_tbin_v1', 'local_tbin_v2', 'local_tbin_v3'):
        adaptive_signal_window = LocalTBinSignalWindow(
            adaptive_update_kimg,
            num_bins=loss_fn.schedule.num_bins,
            start_nimg=cur_nimg,
        )
    else:
        adaptive_signal_window = None
    if adaptive_signal_window is not None and resume_state_dump:
        if resumed_adaptive_signal_window_state is None:
            raise RuntimeError(
                f'resume training-state missing adaptive_signal_window_state: {resume_state_dump}; '
                'cannot exactly resume adaptive loss aggregation'
            )
        adaptive_signal_window.load_state_dict(
            local_adaptive_signal_window_state(resumed_adaptive_signal_window_state)
        )
        if adaptive_signal_window.next_update_nimg <= cur_nimg:
            raise RuntimeError(
                'resumed adaptive signal window is due before or at the restored progress: '
                f'{adaptive_signal_window.next_update_nimg} <= {cur_nimg}'
            )

    if dist.get_rank() == 0:
        summary_path = os.path.join(run_dir, 'train_summary.csv')
        summary_exists = os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
        if resume_state_dump:
            if summary_exists:
                rows, migrated_backup = load_and_migrate_train_summary(summary_path)
                if migrated_backup is not None:
                    dist.print0(
                        f'Migrated legacy train_summary.csv to telemetry schema; '
                        f'original saved as "{migrated_backup}"'
                    )
                last = rows[-1]
                last_attempted = int(float(last['attempted_iteration']))
                last_nimg = int(float(last.get('processed_nimg', last.get('nimg', -1))))
                last_schedule = str(last.get('schedule', '')).strip()
                if last_schedule and last_schedule != str(schedule_name):
                    raise RuntimeError(
                        f'train_summary.csv schedule={last_schedule!r} does not match '
                        f'current schedule={schedule_name!r}; refuse mixed-schedule resume'
                    )
                if attempted_iteration and last_attempted != attempted_iteration:
                    raise RuntimeError(
                        f'train_summary.csv last attempted_iteration={last_attempted} '
                        f'does not match training-state attempted_iteration={attempted_iteration}'
                    )
                if last_nimg >= 0 and last_nimg != cur_nimg:
                    raise RuntimeError(
                        f'train_summary.csv last processed_nimg={last_nimg} '
                        f'does not match resumed cur_nimg={cur_nimg}'
                    )
                last_next_loop_tick = str(last.get('next_loop_cur_tick', '')).strip()
                if last_next_loop_tick:
                    try:
                        parsed_next_loop_tick = float(last_next_loop_tick)
                    except ValueError as exc:
                        raise RuntimeError(
                            'train_summary.csv last next_loop_cur_tick must be numeric: '
                            f'{last_next_loop_tick!r}'
                        ) from exc
                    if (
                        not math.isfinite(parsed_next_loop_tick)
                        or not parsed_next_loop_tick.is_integer()
                        or parsed_next_loop_tick < 0
                    ):
                        raise RuntimeError(
                            'train_summary.csv last next_loop_cur_tick must be a '
                            f'non-negative integer: {last_next_loop_tick!r}'
                        )
                    if int(parsed_next_loop_tick) != cur_tick:
                        raise RuntimeError(
                            f'train_summary.csv last next_loop_cur_tick={last_next_loop_tick} '
                            f'does not match resumed cur_tick={cur_tick}'
                        )
                if not attempted_iteration:
                    attempted_iteration = last_attempted
                    successful_optimizer_steps = int(float(
                        last.get('successful_optimizer_steps', last_attempted)
                    ))
            train_summary_csv = open(summary_path, 'at', newline='')
            train_summary_writer = csv.DictWriter(train_summary_csv, fieldnames=_TRAIN_SUMMARY_FIELDS)
            if not summary_exists:
                train_summary_writer.writeheader()
                train_summary_csv.flush()
        else:
            if summary_exists:
                raise RuntimeError(
                    f'fresh run refuses to append existing train_summary.csv: {summary_path}; '
                    f'pass --resume for a legal continuation or use an empty outdir'
                )
            train_summary_csv = open(summary_path, 'wt', newline='')
            train_summary_writer = csv.DictWriter(train_summary_csv, fieldnames=_TRAIN_SUMMARY_FIELDS)
            train_summary_writer.writeheader()
            train_summary_csv.flush()

    factorial_telemetry_csv = None
    factorial_telemetry_writer = None
    if strict_reproducibility and dist.get_rank() == 0:
        telemetry_path = os.path.join(
            run_dir, 'factorial_training_telemetry_v1.csv'
        )
        telemetry_exists = (
            os.path.isfile(telemetry_path)
            and os.path.getsize(telemetry_path) > 0
        )
        if resume_state_dump:
            if not telemetry_exists:
                raise RuntimeError(
                    'strict factorial resume requires existing versioned telemetry'
                )
            with open(telemetry_path, 'rt', newline='') as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != _FACTORIAL_TELEMETRY_FIELDS:
                    raise RuntimeError(
                        'factorial telemetry schema does not match v1 exactly'
                    )
                rows = list(reader)
            if not rows:
                raise RuntimeError('factorial telemetry has no attempted rows')
            last = rows[-1]
            if int(last['attempted_iteration']) != attempted_iteration:
                raise RuntimeError(
                    'factorial telemetry attempt does not match training-state'
                )
            if int(last['processed_nimg']) != cur_nimg:
                raise RuntimeError(
                    'factorial telemetry nimg does not match training-state'
                )
            if last['arm'] != loss_fn.factorial['arm']:
                raise RuntimeError(
                    'factorial telemetry arm does not match current config'
                )
            factorial_telemetry_csv = open(
                telemetry_path, 'at', newline=''
            )
        else:
            if telemetry_exists:
                raise RuntimeError(
                    'fresh run refuses existing factorial telemetry'
                )
            factorial_telemetry_csv = open(
                telemetry_path, 'xt', newline=''
            )
        factorial_telemetry_writer = csv.DictWriter(
            factorial_telemetry_csv,
            fieldnames=_FACTORIAL_TELEMETRY_FIELDS,
        )
        if not resume_state_dump:
            factorial_telemetry_writer.writeheader()
            factorial_telemetry_csv.flush()

    # Prepare for the mapping fn p(r|t).
    dist.print0(f'Reduce dt every {double_ticks} ticks.')
    
    def update_scheduler(loss_fn):
        loss_fn.update_schedule(stage)
        dist.print0(f'Update scheduler at {cur_tick} ticks, {cur_nimg / 1e3} kimg, ratio {loss_fn.ratio}')

    def build_training_state(
        adaptive_signal_window_state=None,
        rank_states=None,
        *,
        advance_tick=True,
    ):
        # Natural maintenance checkpoints are written before the loop advances
        # cur_tick and tick_start_nimg, so persist their next-loop values.  A
        # planned-pause-only checkpoint is merely a durability boundary: it
        # must preserve the current tick controls exactly so resume matches a
        # run that never paused.
        data = dict(
            net=net,
            optimizer_state=optimizer.state_dict(),
            attempted_iteration=attempted_iteration,
            successful_optimizer_steps=successful_optimizer_steps,
            cur_nimg=cur_nimg,
            cur_tick=cur_tick + int(advance_tick),
            tick_start_nimg=(cur_nimg if advance_tick else tick_start_nimg),
            # Match the final CSV row exactly; resume timing continues from
            # the last completed attempted iteration rather than from later
            # checkpoint I/O and maintenance work.
            elapsed_sec=elapsed_sec,
        )
        if hasattr(loss_fn, 'schedule_state_dict'):
            data['loss_fn_state'] = loss_fn.schedule_state_dict()
        if adaptive_signal_window is not None:
            if adaptive_signal_window_state is None:
                raise RuntimeError('adaptive signal window state was not collected for checkpointing')
            data['adaptive_signal_window_state'] = adaptive_signal_window_state
        if enable_amp:
            data['gradscaler_state'] = scaler.state_dict()
        if power_ema is not None:
            data['power_ema_state'] = power_ema.state_dict()
        if exact_reproducibility:
            if rank_states is None:
                raise RuntimeError(
                    'exact state requires every rank RNG/sampler state'
                )
            next_cur_tick = cur_tick + int(advance_tick)
            next_stage = max((next_cur_tick - 1) // double_ticks, 0)
            strict_loss_state = dict(data['loss_fn_state'])
            strict_loss_state.update(
                stage=next_stage,
                ratio=1 - 1 / loss_fn.q ** (next_stage + 1),
            )
            data.update(
                reproducibility_schema=(
                    reproducibility.TRAINING_STATE_SCHEMA
                    if strict_reproducibility
                    else reproducibility.EXACT_TRAINING_STATE_SCHEMA
                ),
                ema=ema,
                rank_states=rank_states,
                trajectory_config=strict_trajectory_config,
                trajectory_config_sha256=strict_trajectory_config_sha256,
                loss_fn_state=strict_loss_state,
                snapshot_grid_z=[value.detach().cpu() for value in grid_z],
                snapshot_grid_c=[value.detach().cpu() for value in grid_c],
                snapshot_grid_size=tuple(grid_size),
            )
            if strict_reproducibility:
                data['factorial'] = dict(loss_fn.factorial)
        return data
        
    # cur_tick in a checkpoint denotes the next loop.  The uninterrupted loop
    # updates its stage from (cur_tick - 1) after natural maintenance.
    stage = max((cur_tick - 1) // double_ticks, 0)
    if exact_reproducibility and resume_state_dump:
        if loss_fn.stage != stage:
            raise RuntimeError(
                'exact restored loss stage does not match tick state'
            )
    update_scheduler(loss_fn)

    initial_receipt_path = (
        os.path.join(run_dir, 'initial_state_receipt_v1.json')
        if dist.get_rank() == 0 else None
    )
    if strict_reproducibility and resume_state_dump:
        if dist.get_rank() == 0 and not os.path.isfile(initial_receipt_path):
            raise RuntimeError(
                'strict factorial resume requires the original initial-state receipt'
            )
    elif strict_reproducibility:
        initial_rank_states = gather_rank_reproducibility_state(
            dataset_sampler, local_consumed_samples
        )
        if dist.get_rank() == 0:
            model_sha256 = reproducibility.module_state_sha256(net)
            ema_sha256 = reproducibility.module_state_sha256(ema)
            optimizer_sha256 = reproducibility.state_sha256(
                optimizer.state_dict()
            )
            gradscaler_sha256 = reproducibility.state_sha256(
                scaler.state_dict() if enable_amp else None
            )
            rank_receipts = [
                {
                    'rank': state['rank'],
                    'world_size': state['world_size'],
                    'rng_sha256': reproducibility.state_sha256(
                        state['rng_state']
                    ),
                    'sampler_sha256': reproducibility.state_sha256(
                        state['sampler_state']
                    ),
                    'sampler_state': state['sampler_state'],
                }
                for state in initial_rank_states
            ]
            common_hashes = {
                'model': model_sha256,
                'ema': ema_sha256,
                'optimizer': optimizer_sha256,
                'gradscaler': gradscaler_sha256,
                'rank_rng': [row['rng_sha256'] for row in rank_receipts],
                'rank_sampler': [
                    row['sampler_sha256'] for row in rank_receipts
                ],
            }
            reproducibility.atomic_json_dump(
                {
                    'schema': reproducibility.INITIAL_RECEIPT_SCHEMA,
                    'seed': seed,
                    'attempted_iteration': attempted_iteration,
                    'processed_nimg': cur_nimg,
                    'factorial': dict(loss_fn.factorial),
                    'dataset_path': dataset_kwargs.get('path'),
                    'transfer_path': resume_pkl,
                    'world_size': dist.get_world_size(),
                    'batch_size': batch_size,
                    'batch_gpu': batch_gpu,
                    'trajectory_config': strict_trajectory_config,
                    'trajectory_config_sha256': (
                        strict_trajectory_config_sha256
                    ),
                    'hashes': common_hashes,
                    'common_initial_state_sha256': (
                        reproducibility.state_sha256(common_hashes)
                    ),
                    'rank_states': rank_receipts,
                },
                initial_receipt_path,
                overwrite=False,
            )

    if (
        stop_after_attempts is not None
        and attempted_iteration >= stop_after_attempts
    ):
        raise RuntimeError(
            'planned pause target must be greater than restored attempts'
        )

    # Already at/past the requested budget (e.g. resume with same duration): do not
    # execute an extra optimizer step before noticing done.
    if cur_nimg >= total_kimg * 1000:
        dist.print0(f'Already reached training budget at {cur_nimg / 1e3:.3f} kimg; exiting.')
        if train_summary_csv is not None:
            train_summary_csv.close()
        if factorial_telemetry_csv is not None:
            factorial_telemetry_csv.close()
        dist.print0()
        dist.print0('Exiting...')
        return

    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        loss_batches = []
        schedule_metric_batches = []
        local_signal_batches = []
        factorial_metric_batches = []
        batch_input_hashes = []
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                local_consumed_samples += int(images.shape[0])
                if strict_reproducibility:
                    batch_input_hashes.append(
                        reproducibility.state_sha256({
                            'images': images,
                            'labels': labels,
                        })
                    )
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                loss_batches.append(loss.detach())
                schedule_metric_batches.append(loss_fn.schedule_runtime_metrics())
                if strict_reproducibility:
                    factorial_metric_batches.append(
                        loss_fn.factorial_runtime_metrics()
                    )
                if schedule_name in ('local_tbin_v1', 'local_tbin_v2', 'local_tbin_v3'):
                    signal = loss_fn.local_training_signal()
                    if signal is None:
                        raise RuntimeError(
                            f'{schedule_name} did not produce raw per-bin signal'
                        )
                    local_signal_batches.append(signal)
                training_stats.report('Loss/loss', loss)
                if enable_amp:
                    if global_batch_mean:
                        scaler.scale(
                            loss.mean() / num_accumulation_rounds
                        ).backward()
                    else:
                        scaler.scale(loss.mean()).backward()
                else:
                    if global_batch_mean:
                        loss.mul(
                            loss_scaling / num_accumulation_rounds
                        ).mean().backward()
                    else:
                        loss.mul(loss_scaling).mean().backward()

        # Unscale first so GradScaler can detect non-finite gradients before
        # they are sanitized below. scaler.step() will still skip the update
        # when unscale_() records an overflow.
        if enable_amp:
            scaler.unscale_(optimizer)

        if strict_reproducibility:
            loss_nonfinite_count, _, _ = tensor_collection_diagnostics(
                loss_batches
            )
            (
                raw_grad_nonfinite_count,
                raw_grad_norm,
                raw_grad_finite_norm,
            ) = tensor_collection_diagnostics(
                param.grad for param in net.parameters()
                if param.grad is not None
            )
            parameters_before_step = [
                param.detach().clone() for param in net.parameters()
            ]
        elif generic_exact_resume:
            loss_nonfinite_count, raw_grad_nonfinite_count = (
                enforce_generic_exact_finite_before_sanitization(
                    loss_batches, net.parameters(), device
                )
            )

        # NOTE(aiihn & Gsunshine): This should be further tested for AMP.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        if strict_reproducibility:
            (
                sanitized_grad_nonfinite_count,
                sanitized_grad_norm,
                _,
            ) = tensor_collection_diagnostics(
                param.grad for param in net.parameters()
                if param.grad is not None
            )

        if lr_kwargs is not None:
            learning_rate = learning_rate_schedule(
                cur_nimg=cur_nimg,
                batch_size=batch_size,
                rampup_kimg=lr_rampup_kimg,
                **lr_kwargs,
            )
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

        # Update weights. Record GradScaler scale / skip for train_summary.csv.
        # scale_before is the scale applied to this step; a drop after update()
        # means overflow was detected and optimizer.step was skipped.
        grad_scale = float(loss_scaling)
        scale_after = grad_scale
        step_skipped = 0
        if enable_amp:
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            grad_scale = scale_before
            step_skipped = int(scale_after < scale_before)
        else:
            optimizer.step()

        if strict_reproducibility:
            (
                update_nonfinite_count,
                update_norm,
                _,
            ) = tensor_collection_diagnostics(
                param.detach() - before
                for param, before in zip(
                    net.parameters(), parameters_before_step
                )
            )
            del parameters_before_step
            (
                model_nonfinite_count,
                model_norm,
                _,
            ) = tensor_collection_diagnostics(net.parameters())
        elif generic_exact_resume:
            model_nonfinite_count = tensor_collection_nonfinite_count(
                tensor for _, tensor in misc.named_params_and_buffers(net)
            )

        attempted_iteration += 1
        if not step_skipped:
            successful_optimizer_steps += 1

        loss_count = sum(x.numel() for x in loss_batches)
        loss_sum = sum(float(x.sum().cpu()) for x in loss_batches)
        loss_mean = loss_sum / loss_count
        runtime_pair_metrics = globally_average_runtime_pairs(schedule_metric_batches, device=device)
        elapsed_sec = elapsed_base_sec + (time.time() - start_time)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
        training_stats.report0('Progress/grad_scale', grad_scale)
        training_stats.report0('Progress/step_skipped', step_skipped)
        training_stats.report0('Progress/attempted_iteration', attempted_iteration)
        training_stats.report0('Progress/successful_optimizer_steps', successful_optimizer_steps)
        training_stats.report0('Timing/elapsed_sec', elapsed_sec)
        training_stats.report0('Resources/update_peak_gpu_mem_gb', peak_vram_gb)

        # Update EMA.
        if ema_halflife_kimg is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        if strict_reproducibility:
            (
                ema_nonfinite_count,
                ema_norm,
                _,
            ) = tensor_collection_diagnostics(ema.parameters())
        elif generic_exact_resume:
            ema_nonfinite_count = tensor_collection_nonfinite_count(
                tensor for _, tensor in misc.named_params_and_buffers(ema)
            )

        # Advance iteration-local state. Adaptive updates intentionally happen
        # here, before the maintenance early-continue below.
        cur_nimg += batch_size
        power_ema_nonfinite_count = 0
        if power_ema is not None:
            power_ema.update(cur_nimg=cur_nimg, batch_size=batch_size)
            if generic_exact_resume:
                power_ema_nonfinite_count = tensor_collection_nonfinite_count(
                    tensor
                    for profile in power_ema.emas
                    for _, tensor in misc.named_params_and_buffers(profile)
                )
        if generic_exact_resume:
            (
                model_nonfinite_count,
                ema_nonfinite_count,
                power_ema_nonfinite_count,
            ) = globally_sum_counts(
                (
                    model_nonfinite_count,
                    ema_nonfinite_count,
                    power_ema_nonfinite_count,
                ),
                device,
            )
            exact_state_diagnostics = {
                'optimizer update/model': model_nonfinite_count,
                'EMA': ema_nonfinite_count,
            }
            if power_ema is not None:
                exact_state_diagnostics['PowerEMA'] = power_ema_nonfinite_count
            enforce_generic_exact_finite(
                'state update', exact_state_diagnostics
            )
        if adaptive_signal_window is not None:
            if isinstance(adaptive_signal_window, LocalTBinSignalWindow):
                local_sums = torch.stack(
                    [batch['loss_sums'] for batch in local_signal_batches]
                ).sum(dim=0)
                local_counts = torch.stack(
                    [batch['loss_counts'] for batch in local_signal_batches]
                ).sum(dim=0)
                adaptive_signal_window.add(local_sums.tolist(), local_counts.tolist())
                signal_window = adaptive_signal_window.pop_if_due(cur_nimg)
                if signal_window is not None:
                    signal_loss = globally_average_local_tbin_loss(
                        *signal_window, device=device
                    )
                    loss_fn.update_training_signal(signal_loss)
            else:
                adaptive_signal_window.add(loss_sum, loss_count)
                signal_window = adaptive_signal_window.pop_if_due(cur_nimg)
                if signal_window is not None:
                    signal_loss = globally_average_adaptive_loss(*signal_window, device=device)
                    loss_fn.update_training_signal(signal_loss)

        schedule_runtime_metrics = loss_fn.schedule_runtime_metrics()
        schedule_runtime_metrics.update(runtime_pair_metrics)
        if schedule_runtime_metrics['loss_ema'] is not None:
            training_stats.report0('Schedule/loss_ema', schedule_runtime_metrics['loss_ema'])
        if schedule_runtime_metrics['loss_reference'] is not None:
            training_stats.report0('Schedule/loss_reference', schedule_runtime_metrics['loss_reference'])
        training_stats.report0('Schedule/correction', schedule_runtime_metrics['correction'])
        training_stats.report0('Schedule/signal_updates', schedule_runtime_metrics['signal_updates'])
        training_stats.report0('Schedule/adaptive_active', int(schedule_runtime_metrics['adaptive_active']))
        training_stats.report0('Schedule/r_over_t_mean', schedule_runtime_metrics['r_over_t_mean'])
        training_stats.report0('Schedule/gap_mean', schedule_runtime_metrics['gap_mean'])
        training_stats.report0(
            'Schedule/gap_over_sigmoid_gap_mean',
            schedule_runtime_metrics['gap_over_sigmoid_gap_mean'],
        )
        training_stats.report0(
            'Schedule/lower_gap_clip_rate',
            schedule_runtime_metrics['lower_gap_clip_rate'],
        )
        training_stats.report0(
            'Schedule/upper_gap_clip_rate',
            schedule_runtime_metrics['upper_gap_clip_rate'],
        )
        local_runtime_metrics = loss_fn.schedule_local_runtime_metrics()
        if local_runtime_metrics is not None:
            for index in range(len(local_runtime_metrics['gap_scales'])):
                prefix = f'Schedule/tbin{index}'
                for key in ('last_raw_loss', 'short_ema', 'long_ema'):
                    value = local_runtime_metrics[key][index]
                    if value is not None:
                        training_stats.report0(f'{prefix}/{key}', value)
                training_stats.report0(
                    f'{prefix}/gap_scale', local_runtime_metrics['gap_scales'][index]
                )
                training_stats.report0(
                    f'{prefix}/updates', local_runtime_metrics['bin_updates'][index]
                )
                training_stats.report0(
                    f'{prefix}/active', int(local_runtime_metrics['bin_active'][index])
                )

        if strict_reproducibility:
            factorial_metrics = aggregate_factorial_runtime_metrics(
                factorial_metric_batches
            )
            learning_rates = {
                float(group['lr']) for group in optimizer.param_groups
            }
            if len(learning_rates) != 1:
                raise RuntimeError(
                    'strict factorial protocol requires one common learning rate'
                )
            learning_rate = learning_rates.pop()
            telemetry_schema = (
                'ect.q128.matched-spacing-training-telemetry/v1'
                if factorial_metrics['protocol'] == 'q128_matched_spacing_v1'
                else 'ect.q256.target-weight-training-telemetry/v1'
            )
            telemetry_row = {
                'schema': telemetry_schema,
                'protocol': factorial_metrics['protocol'],
                'arm': factorial_metrics['arm'],
                'target_gap_scale': f"{factorial_metrics['target_gap_scale']:.17g}",
                'denominator_gap_scale': f"{factorial_metrics['denominator_gap_scale']:.17g}",
                'attempted_iteration': attempted_iteration,
                'successful_optimizer_steps': successful_optimizer_steps,
                'processed_nimg': cur_nimg,
                'processed_kimg': f'{cur_nimg / 1e3:.6f}',
                'stage': stage,
                'loss': f'{loss_mean:.17g}',
                'loss_nonfinite_count': loss_nonfinite_count,
                'raw_grad_norm': f'{raw_grad_norm:.17g}',
                'raw_grad_finite_norm': f'{raw_grad_finite_norm:.17g}',
                'raw_grad_nonfinite_count': raw_grad_nonfinite_count,
                'sanitized_grad_norm': f'{sanitized_grad_norm:.17g}',
                'sanitized_grad_nonfinite_count': (
                    sanitized_grad_nonfinite_count
                ),
                'update_norm': f'{update_norm:.17g}',
                'update_nonfinite_count': update_nonfinite_count,
                'model_norm': f'{model_norm:.17g}',
                'model_nonfinite_count': model_nonfinite_count,
                'ema_norm': f'{ema_norm:.17g}',
                'ema_nonfinite_count': ema_nonfinite_count,
                'sample_count': factorial_metrics['sample_count'],
                'batch_sha256': reproducibility.state_sha256(
                    batch_input_hashes
                ),
                't_sha256': factorial_metrics['t_sha256'],
                'base_r_sha256': factorial_metrics['base_r_sha256'],
                'target_r_sha256': factorial_metrics['target_r_sha256'],
                'denominator_r_sha256': factorial_metrics['denominator_r_sha256'],
                'target_delta_sha256': factorial_metrics['target_delta_sha256'],
                'denominator_delta_sha256': factorial_metrics['denominator_delta_sha256'],
                'base_r_zero_count': factorial_metrics['base_r_zero_count'],
                'target_r_zero_count': factorial_metrics['target_r_zero_count'],
                'target_r_equal_t_count': factorial_metrics['target_r_equal_t_count'],
                'target_scaled_to_zero_count': factorial_metrics['target_scaled_to_zero_count'],
                'denominator_r_zero_count': factorial_metrics['denominator_r_zero_count'],
                'denominator_r_equal_t_count': factorial_metrics['denominator_r_equal_t_count'],
                'denominator_scaled_to_zero_count': factorial_metrics['denominator_scaled_to_zero_count'],
                'target_delta_min': f"{factorial_metrics['target_delta_min']:.17g}",
                'target_delta_max': f"{factorial_metrics['target_delta_max']:.17g}",
                'target_delta_mean': f"{factorial_metrics['target_delta_mean']:.17g}",
                'denominator_delta_min': f"{factorial_metrics['denominator_delta_min']:.17g}",
                'denominator_delta_max': f"{factorial_metrics['denominator_delta_max']:.17g}",
                'denominator_delta_mean': f"{factorial_metrics['denominator_delta_mean']:.17g}",
                'factor_nonfinite_count': factorial_metrics['nonfinite_count'],
                'nonpositive_denominator_count': factorial_metrics['nonpositive_denominator_count'],
                'learning_rate': f'{learning_rate:.17g}',
                'grad_scale_before': f'{grad_scale:.17g}',
                'grad_scale_after': f'{scale_after:.17g}',
                'step_skipped': step_skipped,
                'elapsed_sec': f'{elapsed_sec:.6f}',
                'gpu_hours_cumulative': f'{elapsed_sec / 3600:.9f}',
            }
            if factorial_telemetry_writer is not None:
                factorial_telemetry_writer.writerow(telemetry_row)
                factorial_telemetry_csv.flush()

            invariant_failures = []
            if factorial_metrics['sample_count'] != batch_size:
                invariant_failures.append('factorial sample_count != batch_size')
            expected_local_consumed = cur_nimg // dist.get_world_size()
            if local_consumed_samples != expected_local_consumed:
                invariant_failures.append('sampler consumption != processed_nimg')
            if loss_nonfinite_count:
                invariant_failures.append('non-finite loss')
            if sanitized_grad_nonfinite_count:
                invariant_failures.append('non-finite sanitized gradient')
            if update_nonfinite_count or model_nonfinite_count or ema_nonfinite_count:
                invariant_failures.append('non-finite update/model/EMA')
            if factorial_metrics['nonfinite_count']:
                invariant_failures.append('non-finite target/denominator factor')
            if factorial_metrics['nonpositive_denominator_count']:
                invariant_failures.append('non-positive realized denominator')
            if bool(raw_grad_nonfinite_count) != bool(step_skipped):
                invariant_failures.append(
                    'raw gradient non-finite status does not match AMP skip'
                )
            if step_skipped and update_norm != 0:
                invariant_failures.append('skipped optimizer attempt changed parameters')
            if not step_skipped and (not math.isfinite(update_norm) or update_norm <= 0):
                invariant_failures.append('successful optimizer update norm is not positive')
            if invariant_failures:
                if factorial_telemetry_csv is not None:
                    os.fsync(factorial_telemetry_csv.fileno())
                raise FloatingPointError(
                    'strict factorial training invariant failure: '
                    + '; '.join(invariant_failures)
                )

        # Record the exact state that the following loop iteration will see.
        # This cannot be derived reliably from image count: the first iteration
        # always performs maintenance, and completion forces it regardless of
        # --tick. A checkpoint saved below persists this same cur_tick value.
        done = (cur_nimg >= total_kimg * 1000)
        planned_pause = (
            stop_after_attempts is not None
            and attempted_iteration >= stop_after_attempts
            and not done
        )
        natural_maintenance_due = (
            done
            or cur_tick == 0
            or cur_nimg >= tick_start_nimg + kimg_per_tick * 1000
        )
        maintenance_due = natural_maintenance_due or planned_pause
        next_loop_cur_tick = cur_tick + int(natural_maintenance_due)

        if train_summary_writer is not None:
            train_summary_writer.writerow({
                'attempted_iteration': attempted_iteration,
                'successful_optimizer_steps': successful_optimizer_steps,
                'processed_nimg': cur_nimg,
                'processed_kimg': f'{cur_nimg / 1e3:.6f}',
                'loss': f'{loss_mean:.8f}',
                'grad_scale': f'{grad_scale:.8g}',
                'step_skipped': step_skipped,
                'schedule': schedule_name,
                'stage': stage,
                'next_loop_cur_tick': next_loop_cur_tick,
                'loss_ema': '' if schedule_runtime_metrics['loss_ema'] is None else f"{schedule_runtime_metrics['loss_ema']:.12g}",
                'loss_reference': '' if schedule_runtime_metrics['loss_reference'] is None else f"{schedule_runtime_metrics['loss_reference']:.12g}",
                'correction': f"{schedule_runtime_metrics['correction']:.12g}",
                'signal_updates': schedule_runtime_metrics['signal_updates'],
                'adaptive_active': int(schedule_runtime_metrics['adaptive_active']),
                'r_over_t_mean': f"{schedule_runtime_metrics['r_over_t_mean']:.12g}",
                'gap_mean': f"{schedule_runtime_metrics['gap_mean']:.12g}",
                'gap_over_sigmoid_gap_mean': f"{schedule_runtime_metrics['gap_over_sigmoid_gap_mean']:.12g}",
                'lower_gap_clip_rate': f"{schedule_runtime_metrics['lower_gap_clip_rate']:.12g}",
                'upper_gap_clip_rate': f"{schedule_runtime_metrics['upper_gap_clip_rate']:.12g}",
                'elapsed_sec': f'{elapsed_sec:.6f}',
                'peak_vram_gb': f'{peak_vram_gb:.6f}',
            })
            train_summary_csv.flush()

        immutable_checkpoint_due = cur_nimg in immutable_checkpoint_nimg
        if immutable_checkpoint_due:
            if train_summary_csv is not None:
                train_summary_csv.flush()
                os.fsync(train_summary_csv.fileno())
            if factorial_telemetry_csv is not None:
                factorial_telemetry_csv.flush()
                os.fsync(factorial_telemetry_csv.fileno())
            immutable_adaptive_state = None
            if adaptive_signal_window is not None:
                immutable_adaptive_state = (
                    gather_adaptive_signal_window_state(
                        adaptive_signal_window, device
                    )
                )
            immutable_rank_states = None
            if exact_reproducibility:
                expected_local_consumed = cur_nimg // dist.get_world_size()
                if local_consumed_samples != expected_local_consumed:
                    raise RuntimeError(
                        'immutable checkpoint sampler consumption does not '
                        'match per-rank processed images'
                    )
                immutable_rank_states = gather_rank_reproducibility_state(
                    dataset_sampler,
                    local_consumed_samples,
                    device=(device if generic_exact_resume else None),
                )
            if dist.get_rank() == 0:
                immutable_path = save_immutable_training_state(
                    build_training_state(
                        immutable_adaptive_state,
                        immutable_rank_states,
                        advance_tick=natural_maintenance_due,
                    ),
                    run_dir,
                    cur_nimg,
                )
                dist.print0(
                    'Saved immutable full-state milestone at '
                    f'{cur_nimg / 1000:.3f} kimg: {immutable_path}'
                )

        # Perform maintenance tasks once per tick.
        if not maintenance_due:
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<9.5f}"]
        fields += [f"grad_scale {grad_scale:<9g}"]
        fields += [f"step_skipped {step_skipped:<7d}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        state_dump_due = (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
        )
        latest_checkpoint_due = (
            (ckpt_ticks is not None)
            and (done or planned_pause or cur_tick % ckpt_ticks == 0)
            and cur_tick != 0
        )
        checkpoint_rank_states = None
        checkpoint_adaptive_state = None
        if state_dump_due or latest_checkpoint_due:
            if train_summary_csv is not None:
                train_summary_csv.flush()
                os.fsync(train_summary_csv.fileno())
            if factorial_telemetry_csv is not None:
                factorial_telemetry_csv.flush()
                os.fsync(factorial_telemetry_csv.fileno())
            if adaptive_signal_window is not None:
                checkpoint_adaptive_state = (
                    gather_adaptive_signal_window_state(
                        adaptive_signal_window, device
                    )
                )
            if exact_reproducibility:
                if cur_nimg % dist.get_world_size() != 0:
                    raise RuntimeError(
                        'exact cur_nimg is not divisible by world size'
                    )
                expected_local_consumed = cur_nimg // dist.get_world_size()
                if local_consumed_samples != expected_local_consumed:
                    raise RuntimeError(
                        'logical sampler consumption does not match committed '
                        f'progress: {local_consumed_samples} != '
                        f'{expected_local_consumed}'
                    )
                checkpoint_rank_states = gather_rank_reproducibility_state(
                    dataset_sampler,
                    local_consumed_samples,
                    device=(device if generic_exact_resume else None),
                )

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick != 0:
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                reproducibility.atomic_pickle_dump(
                    data,
                    os.path.join(
                        run_dir, f'network-snapshot-{cur_tick:06d}.pkl'
                    ),
                    overwrite=False,
                )
            del data # conserve memory

        # Save full dump of the training state. Every rank participates in
        # collecting its local adaptive-loss accumulator; rank 0 writes the
        # resulting combined state.
        if state_dump_due:
            if dist.get_rank() == 0:
                reproducibility.atomic_torch_save(
                    build_training_state(
                        checkpoint_adaptive_state,
                        checkpoint_rank_states,
                        advance_tick=natural_maintenance_due,
                    ),
                    os.path.join(run_dir, f'training-state-{cur_tick:06d}.pt'),
                    overwrite=False,
                )

        # Save latest checkpoints
        if latest_checkpoint_due:
            dist.print0(f'Save the latest checkpoint at {cur_tick:06d} img...')
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                reproducibility.atomic_pickle_dump(
                    data,
                    os.path.join(run_dir, 'network-snapshot-latest.pkl'),
                    overwrite=True,
                )
            del data # conserve memory

            if dist.get_rank() == 0:
                reproducibility.atomic_torch_save(
                    build_training_state(
                        checkpoint_adaptive_state,
                        checkpoint_rank_states,
                        advance_tick=natural_maintenance_due,
                    ),
                    os.path.join(run_dir, f'training-state-latest.pt'),
                    overwrite=True,
                )

        # Sample Img
        if (sample_ticks is not None) and (done or cur_tick % sample_ticks == 0) and dist.get_rank() == 0:
            dist.print0('Exporting sample images...')
            images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
            del images
    
        # Evaluation
        if metrics and (eval_ticks is not None) and (done or cur_tick % eval_ticks == 0) and cur_tick > 0:
            dist.print0('Evaluating models...')
            result_dict = metric_main.calc_metric(metric='fid50k_full', 
                    generator_fn=generator_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        
            
            few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
            result_dict = metric_main.calc_metric(metric='two_step_fid50k_full', 
                    generator_fn=few_step_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')                        

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg / 1000, total_kimg)

        # Only a natural maintenance boundary advances tick control state.
        # A planned-pause-only checkpoint must resume from the exact controls
        # that an uninterrupted run would see on its next iteration.
        if natural_maintenance_due:
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        if planned_pause:
            if train_summary_csv is not None:
                train_summary_csv.close()
            if factorial_telemetry_csv is not None:
                factorial_telemetry_csv.close()
            dist.print0()
            dist.print0(
                f'Planned pause after {attempted_iteration} attempts; exiting.'
            )
            return
        
        # Update Scheduler
        new_stage = (cur_tick-1) // double_ticks
        if new_stage > stage:
            stage = new_stage
            update_scheduler(loss_fn)
    
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if sample_ticks is not None and dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'final.png'), drange=[-1,1], grid_size=grid_size)
        del images

    if metrics:
        dist.print0('Evaluating few-step generation...')
        for _ in range(3):
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric,
                    generator_fn=few_step_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl='network-snapshot-latest.pkl')

    # Done.
    if train_summary_csv is not None:
        train_summary_csv.close()
    if factorial_telemetry_csv is not None:
        factorial_telemetry_csv.close()
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
