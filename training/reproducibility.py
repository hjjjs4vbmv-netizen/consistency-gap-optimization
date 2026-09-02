"""Fail-closed state and artifact helpers for exact training continuation."""

import hashlib
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch


RNG_STATE_SCHEMA = 'ect.rank-rng-state/v1'
TRAINING_STATE_SCHEMA = 'ect.q256.target-weight-training-state/v1'
INITIAL_RECEIPT_SCHEMA = 'ect.q256.target-weight-initial-state/v1'
TRAJECTORY_CONFIG_SCHEMA = 'ect.q256.target-weight-trajectory-config/v1'
CURRENT_DEVICE_RNG_STATE_SCHEMA = 'ect.current-device-rng-state/v1'
EXACT_TRAINING_STATE_SCHEMA = 'ect.exact-training-state/v1'
EXACT_TRAJECTORY_CONFIG_SCHEMA = 'ect.exact-trajectory-config/v1'


def canonical_json_data(value):
    """Convert configuration values to one strict JSON-compatible identity."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('configuration floats must be finite')
        return value
    if isinstance(value, np.generic):
        return canonical_json_data(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('configuration dictionary keys must be strings')
        return {
            key: canonical_json_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [canonical_json_data(item) for item in value]
    raise TypeError(
        f'unsupported JSON configuration type: {type(value).__name__}'
    )


def capture_rng_state():
    """Capture every process-local RNG used by the training runtime."""
    return {
        'schema': RNG_STATE_SCHEMA,
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda_all': [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        'torch_cuda_device_count': torch.cuda.device_count()
        if torch.cuda.is_available() else 0,
    }


def restore_rng_state(state):
    """Restore a state produced by :func:`capture_rng_state`."""
    if not isinstance(state, dict) or state.get('schema') != RNG_STATE_SCHEMA:
        raise ValueError('unsupported or missing rank RNG state schema')
    required = ('python', 'numpy', 'torch_cpu', 'torch_cuda_all',
                'torch_cuda_device_count')
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f'rank RNG state missing fields: {", ".join(missing)}')
    expected_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    saved_cuda_count = int(state['torch_cuda_device_count'])
    if saved_cuda_count != expected_cuda_count:
        raise ValueError(
            'CUDA RNG device count mismatch: '
            f'checkpoint={saved_cuda_count}, current={expected_cuda_count}'
        )
    cuda_states = state['torch_cuda_all']
    if not isinstance(cuda_states, list) or len(cuda_states) != saved_cuda_count:
        raise ValueError('CUDA RNG state list length does not match device count')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if saved_cuda_count:
        torch.cuda.set_rng_state_all(cuda_states)


def capture_current_device_rng_state(device):
    """Capture process-local RNG without depending on other visible GPUs."""
    device = torch.device(device)
    cuda_state = None
    if device.type == 'cuda':
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        cuda_state = torch.cuda.get_rng_state(device_index).clone()
    return {
        'schema': CURRENT_DEVICE_RNG_STATE_SCHEMA,
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': cuda_state,
        'torch_cuda_enabled': device.type == 'cuda',
    }


def restore_current_device_rng_state(state, device):
    """Restore the RNG state owned by this process and its training device."""
    if (
        not isinstance(state, dict)
        or state.get('schema') != CURRENT_DEVICE_RNG_STATE_SCHEMA
    ):
        raise ValueError('unsupported or missing current-device RNG state schema')
    required = (
        'python', 'numpy', 'torch_cpu', 'torch_cuda',
        'torch_cuda_enabled',
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(
            f'current-device RNG state missing fields: {", ".join(missing)}'
        )
    device = torch.device(device)
    current_index = None
    if device.type == 'cuda':
        current_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
    if bool(state['torch_cuda_enabled']) != (current_index is not None):
        raise ValueError('CUDA RNG state does not match the current device type')
    if (state['torch_cuda'] is None) == bool(state['torch_cuda_enabled']):
        raise ValueError('current-device CUDA RNG payload is inconsistent')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if current_index is not None:
        torch.cuda.set_rng_state(state['torch_cuda'], current_index)


def _digest_update(digest, value):
    """Deterministically encode nested runtime state into ``digest``."""
    if value is None:
        digest.update(b'N;')
    elif isinstance(value, bool):
        digest.update(b'B1;' if value else b'B0;')
    elif isinstance(value, int):
        digest.update(f'I{value};'.encode('ascii'))
    elif isinstance(value, float):
        digest.update(f'F{value.hex()};'.encode('ascii'))
    elif isinstance(value, str):
        encoded = value.encode('utf-8')
        digest.update(f'S{len(encoded)}:'.encode('ascii'))
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(f'Y{len(value)}:'.encode('ascii'))
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'T{tensor.dtype}:{tuple(tensor.shape)}:'.encode('ascii'))
        # Byte views also cover dtypes (for example bfloat16) that NumPy
        # cannot represent directly. Flatten first so scalar tensors work.
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order='C')
        )
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f'A{array.dtype}:{array.shape}:'.encode('ascii'))
        digest.update(array.tobytes(order='C'))
    elif isinstance(value, np.generic):
        _digest_update(digest, value.item())
    elif isinstance(value, dict):
        digest.update(f'D{len(value)}:'.encode('ascii'))
        keys = sorted(value, key=lambda key: (type(key).__name__, repr(key)))
        for key in keys:
            _digest_update(digest, key)
            _digest_update(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(
            f'{"Q" if isinstance(value, tuple) else "L"}{len(value)}:'.encode('ascii')
        )
        for item in value:
            _digest_update(digest, item)
    else:
        raise TypeError(f'unsupported state-digest type: {type(value).__name__}')


def state_sha256(value):
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def module_state_sha256(module):
    return state_sha256(module.state_dict())


def _fsync_parent(path):
    parent = os.path.dirname(os.path.abspath(path)) or '.'
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, writer, *, overwrite):
    """Write and fsync a file, then publish it atomically.

    Immutable numbered artifacts use an atomic hard-link publication so an
    existing target can never be replaced. Mutable ``latest`` aliases use
    ``os.replace`` but are never observed partially written.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    temporary = f'{path}.tmp-{os.getpid()}-{time.time_ns()}'
    try:
        with open(temporary, 'xb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        _fsync_parent(path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_pickle_dump(value, path, *, overwrite=False):
    _atomic_write(
        path,
        lambda handle: pickle.dump(value, handle),
        overwrite=overwrite,
    )


def atomic_torch_save(value, path, *, overwrite=False):
    _atomic_write(
        path,
        lambda handle: torch.save(value, handle),
        overwrite=overwrite,
    )


def atomic_json_dump(value, path, *, overwrite=False):
    def write(handle):
        payload = json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False
        ).encode('utf-8') + b'\n'
        handle.write(payload)

    _atomic_write(path, write, overwrite=overwrite)
