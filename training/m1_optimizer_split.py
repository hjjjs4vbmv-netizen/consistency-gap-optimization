"""Short, outcome-selected diagnostics; never a formal M1 continuation."""

import json
from pathlib import Path

import torch

from training import loss, m1


def describe(tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                nonfinite=int((~finite).sum()),
                finite_absmax=float(value[finite].abs().max()) if finite.any() else None)


def observe_forward(call, objective, model, images, labels, augment_pipe):
    calls, dtypes = [], []
    module = model.module if hasattr(model, 'module') else model

    def capture(_module, args, _kwargs, output):
        calls.append((args[0].detach().clone(), args[1].detach().clone(), output.detach().clone()))

    def capture_dtype(_module, args):
        dtypes.append(str(args[0].dtype))

    outer = module.register_forward_hook(capture, with_kwargs=True)
    inner = module.model.register_forward_pre_hook(capture_dtype)
    try:
        result = call(objective, net=model, images=images, labels=labels,
                      augment_pipe=augment_pipe)
    finally:
        outer.remove()
        inner.remove()
    if len(calls) != 2:
        raise ValueError('expected both online and target forward calls')
    online, target = calls
    t, r = (entry[1].reshape(-1, 1, 1, 1) for entry in calls)
    raw_target = target[2]
    sanitized = torch.nan_to_num(raw_target)
    target_used = (r > 0) * sanitized + (r <= 0) * images
    residual = online[2] - target_used
    squared = residual ** 2
    reduced = squared.reshape(squared.shape[0], -1).sum(dim=-1)
    # All diagnostic continuations use arm A: target and denominator gaps coincide.
    delta = (t - r).flatten()
    transformed = torch.sqrt(reduced)
    stages = dict(online_output=online[2], target_before_nan_to_num=raw_target,
                  target_after_nan_to_num=sanitized, target_used=target_used,
                  residual=residual, squared=squared, reduced=reduced,
                  sqrt=transformed, delta=delta, divided=transformed / delta,
                  unscaled_loss=result)
    return result, calls, dict(stages={k: describe(v) for k, v in stages.items()},
                               model_input_dtypes=dtypes)


OPERATIONS = ('K', 'R', 'clear_moments', 'reset_step')
SEEDS = (64, 55, 60, 56)


def intervene(optimizer, operation):
    if operation not in OPERATIONS:
        raise ValueError(operation)
    if operation == 'R':
        optimizer.state.clear()
    elif operation != 'K':
        for state in optimizer.state.values():
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError(f'unexpected RAdam state keys: {set(state)}')
            if operation == 'clear_moments':
                state['exp_avg'].zero_()
                state['exp_avg_sq'].zero_()
            else:
                state['step'].zero_()


def state_summary(optimizer):
    states = list(optimizer.state.values())
    return dict(
        step_values=sorted({float(s['step']) for s in states}),
        parameter_states=len(states),
        moment_norms={k: sum(float(s[k].double().square().sum()) for s in states)**.5
                      for k in ('exp_avg', 'exp_avg_sq')})


def install(output, operation):
    output = Path(output)
    original_metadata = m1.initial_metadata

    def metadata(*args, **kwargs):
        return dict(original_metadata(*args, **kwargs),
                    analysis_role='POST_OUTCOME_OPTIMIZER_DIAGNOSTIC',
                    optimizer_operation=operation)

    m1.initial_metadata = metadata
    actual_updates, before = 0, []

    def before_step(optimizer, _args, _kwargs):
        nonlocal before
        if actual_updates == 0:
            before = [p.detach().clone() for g in optimizer.param_groups for p in g['params']]

    def after_step(optimizer, _args, _kwargs):
        nonlocal actual_updates, before
        actual_updates += 1
        if actual_updates == 1:
            parameters = [p for g in optimizer.param_groups for p in g['params']]
            torch.save([p.detach().cpu() - b.cpu() for p, b in zip(parameters, before)],
                       output / 'first_actual_update.pt')
            before = []
        with (output / 'optimizer_steps.jsonl').open('a') as f:
            f.write(json.dumps(dict(actual_update=actual_updates,
                                    step_values=sorted({float(s['step']) for s in optimizer.state.values()})))+'\n')

    def apply(optimizer, branch):
        expected = ('R_' if operation == 'R' else 'K_')
        if not branch.startswith(expected):
            raise ValueError('diagnostic container branch mismatch')
        summary = dict(operation=operation, before=state_summary(optimizer))
        intervene(optimizer, operation)
        summary['after'] = state_summary(optimizer)
        (output / 'intervention.json').write_text(json.dumps(summary, indent=2)+'\n')
        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)
        return int(operation == 'R')

    m1.apply_optimizer_intervention = apply
    original_call = loss.ECMLoss.__call__
    microbatch, saved_failure = 0, False

    def observed(self, net, images, labels=None, augment_pipe=None):
        nonlocal microbatch, saved_failure
        microbatch += 1
        result, _, report = observe_forward(original_call, self, net, images, labels, augment_pipe)
        bad = [k for k, v in report['stages'].items() if v['nonfinite']]
        if microbatch == 1 or (bad and not saved_failure):
            report.update(microbatch=microbatch, attempt=4001+(microbatch-1)//8,
                          operation=operation, first_nonfinite_stage=bad[0] if bad else None)
            name = 'first_forward.json' if microbatch == 1 else 'first_nonfinite_forward.json'
            (output / name).write_text(json.dumps(report, indent=2)+'\n')
        saved_failure |= bool(bad)
        return result

    loss.ECMLoss.__call__ = observed
