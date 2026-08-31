import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import ct_training_loop as training_loop
from training.loss import (
    Q128_MATCHED_SPACING_PROTOCOL,
    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
    resolve_target_weight_factorial,
)
from training import pulse_chase, schedule_switch

STRICT_FACTORIAL_PROTOCOLS = {
    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
    Q128_MATCHED_SPACING_PROTOCOL,
}

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


def parse_immutable_checkpoint_kimg(_ctx, _param, value):
    """Parse a frozen comma-separated list of exact integer kimg budgets."""
    if value is None or value == '':
        return ()
    result = []
    for token in value.split(','):
        token = token.strip()
        if not token or not token.isdigit():
            raise click.BadParameter(
                'expected comma-separated positive integer kimg values'
            )
        budget = int(token)
        if budget <= 0:
            raise click.BadParameter('checkpoint kimg values must be positive')
        result.append(budget)
    if len(set(result)) != len(result):
        raise click.BadParameter('checkpoint kimg values must be unique')
    if result != sorted(result):
        raise click.BadParameter('checkpoint kimg values must be increasing')
    return tuple(result)


def parse_resume_state_token(path):
    match = re.fullmatch(
        r'training-state-(\d+|latest|kimg\d+)\.pt',
        os.path.basename(path),
    )
    return None if match is None else match.group(1)


def normalize_schedule_name(_ctx, _param, value):
    aliases = {
        'adaptive-v1': 'adaptive_v1',
        'global-sigmoid': 'global_sigmoid',
        'local-tbin-v1': 'local_tbin_v1',
        'local-tbin-v2': 'local_tbin_v2',
        'local-tbin-v3': 'local_tbin_v3',
    }
    return aliases.get(value, value)


def make_loss_kwargs(opts):
    """Build the persisted loss/schedule config without renaming legacy keys."""
    kwargs = dnnlib.EasyDict(
        P_mean=opts.mean,
        P_std=opts.std,
        q=opts.q,
        c=opts.c,
        k=opts.k,
        b=opts.b,
        adj=opts.mapping,
        adaptive_loss_ema_beta=opts.adaptive_loss_ema_beta,
        adaptive_warmup_updates=opts.adaptive_warmup_updates,
        adaptive_max_adjust=opts.adaptive_max_adjust,
        adaptive_min_gap=opts.adaptive_min_gap,
        local_tbin_num_bins=opts.local_tbin_num_bins,
        local_tbin_short_beta=opts.local_tbin_short_beta,
        local_tbin_long_beta=opts.local_tbin_long_beta,
        local_tbin_warmup_updates=opts.local_tbin_warmup_updates,
        local_tbin_gain=opts.local_tbin_gain,
        local_tbin_min_scale=opts.local_tbin_min_scale,
        local_tbin_max_scale=opts.local_tbin_max_scale,
        local_tbin_deadband=opts.local_tbin_deadband,
        local_tbin_min_gap=opts.local_tbin_min_gap,
        global_gap_scale=opts.global_gap_scale,
    )
    factorial = resolve_target_weight_factorial(
        getattr(opts, 'factorial_protocol', 'none'),
        getattr(opts, 'target_gap_scale', None),
        getattr(opts, 'denominator_gap_scale', None),
        adj=opts.mapping,
        global_gap_scale=opts.global_gap_scale,
        q=opts.q,
        c=opts.c,
    )
    if factorial['enabled']:
        kwargs.update(
            factorial_protocol=factorial['protocol'],
            target_gap_scale=factorial['target_gap_scale'],
            denominator_gap_scale=factorial['denominator_gap_scale'],
        )
    return kwargs

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='ect',             type=click.Choice(['ect']), default='ect', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--optim',         help='Name of Optimizer', metavar='Optimizer',                     type=str, default='Adam', show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=None, show_default=True)
@click.option('--ema_beta',      help='EMA decay rate', metavar='FLOAT',                            type=click.FloatRange(min=0), default=0.9999, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)

@click.option('--schedule', '--mapping', 'mapping',
              help='Type of t-to-r schedule; --mapping is a compatibility alias', metavar='STR',
              type=click.Choice(['const', 'sigmoid', 'global_sigmoid', 'global-sigmoid',
                                 'adaptive_v1', 'adaptive-v1',
                                 'local_tbin_v1', 'local-tbin-v1',
                                 'local_tbin_v2', 'local-tbin-v2',
                                 'local_tbin_v3', 'local-tbin-v3']),
              callback=normalize_schedule_name, default='sigmoid', show_default=True)
@click.option('--global-gap-scale', help='Fixed multiplier on the official or local sigmoid gap', metavar='FLOAT',
              type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)
@click.option(
    '--factorial-protocol',
    type=click.Choice(['none', *sorted(STRICT_FACTORIAL_PROTOCOLS)]),
    default='none',
    show_default=True,
    help='Enable the frozen q256 target-geometry x denominator protocol',
)
@click.option(
    '--target-gap-scale',
    type=float,
    default=None,
    metavar='FLOAT',
    help='Explicit realized target gap scale for the factorial protocol',
)
@click.option(
    '--denominator-gap-scale',
    type=float,
    default=None,
    metavar='FLOAT',
    help='Explicit realized denominator gap scale for the factorial protocol',
)
@click.option('--adaptive-loss-ema-beta', help='EMA beta for adaptive_v1 loss signal', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, max_open=True), default=0.9, show_default=True)
@click.option('--adaptive-update-kimg', help='Aggregate adaptive_v1 loss signal every KIMG, independent of ticks', metavar='KIMG',
              type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True)
@click.option('--adaptive-warmup-updates', help='Valid adaptive_v1 signal updates before applying corrections', metavar='INT',
              type=click.IntRange(min=0), default=2, show_default=True)
@click.option('--adaptive-max-adjust', help='Maximum absolute adaptive_v1 correction to r/t', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1), default=0.05, show_default=True)
@click.option('--adaptive-min-gap', help='Minimum relative gap (t-r)/t for adaptive_v1', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), default=1e-3, show_default=True)
@click.option('--local-tbin-num-bins', help='Number of p(t)-quantile bins for local t-bin schedules', metavar='INT',
              type=click.IntRange(min=2), default=4, show_default=True)
@click.option('--local-tbin-short-beta', help='Short raw-loss EMA beta for local t-bin schedules', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, max_open=True), default=0.9, show_default=True)
@click.option('--local-tbin-long-beta', help='Long raw-loss EMA beta for local t-bin schedules', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, max_open=True), default=0.99, show_default=True)
@click.option('--local-tbin-warmup-updates', help='Per-bin signal updates before local corrections', metavar='INT',
              type=click.IntRange(min=0), default=32, show_default=True)
@click.option('--local-tbin-gain', help='Trend-to-gap-scale gain for local t-bin schedules', metavar='FLOAT',
              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--local-tbin-min-scale', help='Minimum multiplier on the official sigmoid gap', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, min_open=True), default=0.75, show_default=True)
@click.option('--local-tbin-max-scale', help='Maximum multiplier on the official sigmoid gap', metavar='FLOAT',
              type=click.FloatRange(min=1), default=1.5, show_default=True)
@click.option('--local-tbin-deadband', help='Absolute log-EMA trend ignored by local t-bin schedules', metavar='FLOAT',
              type=click.FloatRange(min=0), default=0.02, show_default=True)
@click.option('--local-tbin-min-gap', help='Minimum relative gap after local scaling', metavar='FLOAT',
              type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), default=1e-3, show_default=True)
@click.option('--double',        help='How often to reduce dt', metavar='TICKS',                    type=click.IntRange(min=1), default=500, show_default=True)

@click.option('-q',              help='Decay Factor', metavar='FLOAT',                              type=click.FloatRange(min=1, min_open=True), default=2.0, show_default=True)
@click.option('-k',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=8.0, show_default=True)
@click.option('-b',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=1.0, show_default=True)

@click.option('-c',              help='Constant c for Adaptive Weighting', metavar='FLOAT',         type=click.FloatRange(), default=0.0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--tf32',          help='Enable tf32 for A100/H100 training speed', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--enable_amp', '--amp', '--enable_gradscaler', 'enable_amp',
              help='Enable torch.cuda.amp.GradScaler; overrides loss scaling set by --ls',
              metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.FloatRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save numbered snapshots; 0 disables them', metavar='TICKS', type=click.IntRange(min=0), default=500, show_default=True)
@click.option('--dump',          help='How often to save numbered state dumps; 0 disables them', metavar='TICKS', type=click.IntRange(min=0), default=500, show_default=True)
@click.option('--ckpt',          help='How often to save latest checkpoints', metavar='TICKS',      type=click.IntRange(min=1), default=50, show_default=True)
@click.option(
    '--immutable-checkpoint-kimg',
    help='Comma-separated exact kimg budgets for immutable full-state saves',
    metavar='KIMG[,KIMG...]',
    type=str,
    callback=parse_immutable_checkpoint_kimg,
    default='',
)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--resume-tick',   help='Number of tick from previous training state', metavar='INT', type=int)
@click.option(
    '--schedule-switch-manifest',
    help='Frozen q256 512-kimg schedule-switch run manifest',
    metavar='JSON',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
)
@click.option(
    '--p2-pulse-chase-manifest',
    help='Frozen q256 B@384 pulse/chase branch manifest',
    metavar='JSON',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
)
@click.option(
    '--p2-matched-randomness-audit',
    is_flag=True,
    help='Smoke-only noise/dropout tape hashing for the P2 randomness gate',
)
@click.option('--stop-after-attempts', help='Gate-only planned pause after N optimizer attempts', metavar='INT', type=click.IntRange(min=1), default=None, hidden=True)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.821]',                             multiple=True, default=[0.821])
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--sample_every',  help='How often to sample imgs', metavar='TICKS',                  type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--eval_every',    help='How often to evaluate metrics', metavar='TICKS',             type=click.IntRange(min=1), default=50, show_default=True)


def main(**kwargs):
    """Train ECMs using the techniques described in the 
    blog "Consistency Models Made Easy".
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = make_loss_kwargs(opts)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name=f'torch.optim.{opts.optim}', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'ect':
        c.network_kwargs.class_name = 'training.networks.ECMPrecond'
        c.loss_kwargs.class_name = 'training.loss.ECMLoss'
    else:
        raise ValueError('Unrecognized Precond & Loss!')

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000) if opts.ema is not None else opts.ema
    c.ema_beta = opts.ema_beta
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench, enable_tf32=opts.tf32, enable_amp=opts.enable_amp)
    c.update(kimg_per_tick=opts.tick,
             snapshot_ticks=None if opts.snap == 0 else opts.snap,
             state_dump_ticks=None if opts.dump == 0 else opts.dump,
             ckpt_ticks=opts.ckpt,
             immutable_checkpoint_kimg=opts.immutable_checkpoint_kimg,
             double_ticks=opts.double, adaptive_update_kimg=opts.adaptive_update_kimg,
             stop_after_attempts=opts.stop_after_attempts,
             schedule_switch_manifest=opts.schedule_switch_manifest,
             pulse_chase_manifest=opts.p2_pulse_chase_manifest,
             matched_randomness_audit=opts.p2_matched_randomness_audit)
    c.update(mid_t=opts.mid_t, metrics=opts.metrics, sample_ticks=opts.sample_every, eval_ticks=opts.eval_every)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        resume_token = parse_resume_state_token(opts.resume)
        if resume_token is None or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        if opts.factorial_protocol in STRICT_FACTORIAL_PROTOCOLS:
            # The versioned factorial training-state is self-contained (net +
            # EMA + optimizer + scaler + RNG + sampler). Depending on a
            # separately replaced `latest` snapshot creates a crash window in
            # which the two files may represent different attempts.
            c.resume_pkl = None
        else:
            c.resume_pkl = os.path.join(
                os.path.dirname(opts.resume),
                f'network-snapshot-{resume_token}.pkl',
            )
        # Prefer explicit --resume-tick; otherwise parse numeric tick from the filename.
        # training-state-latest.pt cannot be converted with int(); the training loop
        # restores the authoritative cur_tick / cur_nimg from the serialized state.
        if opts.resume_tick is not None:
            c.resume_tick = opts.resume_tick
        elif resume_token == 'latest' or resume_token.startswith('kimg'):
            c.resume_tick = 0
        else:
            c.resume_tick = int(resume_token)
        c.resume_state_dump = opts.resume

    if opts.schedule_switch_manifest is not None:
        if opts.resume is None:
            raise click.ClickException(
                '--schedule-switch-manifest requires --resume'
            )
        if opts.factorial_protocol != TARGET_WEIGHT_FACTORIAL_PROTOCOL:
            raise click.ClickException(
                'schedule switch requires q256_target_weight_v1'
            )
        manifest = schedule_switch.load_run_manifest(
            opts.schedule_switch_manifest
        )
        expected = schedule_switch.continuation_factorial(manifest)
        actual = resolve_target_weight_factorial(
            opts.factorial_protocol,
            opts.target_gap_scale,
            opts.denominator_gap_scale,
            adj=opts.mapping,
            global_gap_scale=opts.global_gap_scale,
            q=opts.q,
            c=opts.c,
        )
        if actual != expected:
            raise click.ClickException(
                'CLI factorial scales do not match frozen continuation arm'
            )
        if c.seed != manifest['seed']:
            raise click.ClickException('CLI seed does not match switch manifest')
        if c.total_kimg != manifest['final_kimg']:
            raise click.ClickException(
                'CLI duration does not match switch manifest final budget'
            )
        if c.batch_size != 128 or c.batch_gpu != 16:
            raise click.ClickException(
                'schedule switch requires batch=128 and batch-gpu=16'
            )

    if opts.p2_pulse_chase_manifest is not None:
        if opts.schedule_switch_manifest is not None:
            raise click.ClickException(
                'P2 pulse/chase and legacy schedule-switch manifests are exclusive'
            )
        if opts.resume is None:
            raise click.ClickException('--p2-pulse-chase-manifest requires --resume')
        if opts.factorial_protocol != TARGET_WEIGHT_FACTORIAL_PROTOCOL:
            raise click.ClickException('P2 pulse/chase requires q256_target_weight_v1')
        manifest = pulse_chase.load_run_manifest(opts.p2_pulse_chase_manifest)
        if bool(opts.p2_matched_randomness_audit) != bool(
            manifest.get('matched_randomness_audit')
        ):
            raise click.ClickException(
                'P2 matched-randomness audit flag differs from manifest'
            )
        expected_arm = (
            manifest['pulse_arm'] if c.total_kimg == pulse_chase.PULSE_END_KIMG
            else 'A' if c.total_kimg == pulse_chase.CHASE_END_KIMG
            else None
        )
        if expected_arm is None:
            raise click.ClickException('P2 duration must end at 512 or 640 kimg')
        expected = pulse_chase.factorial_for_arm(expected_arm)
        actual = resolve_target_weight_factorial(
            opts.factorial_protocol,
            opts.target_gap_scale,
            opts.denominator_gap_scale,
            adj=opts.mapping,
            global_gap_scale=opts.global_gap_scale,
            q=opts.q,
            c=opts.c,
        )
        if actual != expected:
            raise click.ClickException('CLI factors do not match frozen P2 phase')
        if c.seed != manifest['seed']:
            raise click.ClickException('CLI seed does not match P2 manifest')
        if c.batch_size != 128 or c.batch_gpu != 16:
            raise click.ClickException('P2 requires batch=128 and batch-gpu=16')
        if os.path.realpath(opts.outdir) != os.path.realpath(
            manifest['immutable_output_root']
        ):
            raise click.ClickException('P2 output directory differs from manifest')
    elif opts.p2_matched_randomness_audit:
        raise click.ClickException(
            '--p2-matched-randomness-audit requires --p2-pulse-chase-manifest'
        )

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-{opts.optim:s}-{opts.lr:f}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        options_path = os.path.join(c.run_dir, 'training_options.json')
        strict_factorial_resume = (
            opts.resume is not None
            and opts.factorial_protocol in STRICT_FACTORIAL_PROTOCOLS
        )
        if opts.p2_pulse_chase_manifest is not None:
            if c.total_kimg == pulse_chase.PULSE_END_KIMG:
                if os.path.exists(options_path):
                    raise click.ClickException(
                        'fresh P2 branch refuses existing training_options.json'
                    )
                target_options_path = options_path
            else:
                if not os.path.isfile(options_path):
                    raise click.ClickException(
                        'P2 chase requires immutable pulse training_options.json'
                    )
                target_options_path = os.path.join(
                    c.run_dir, 'training_options_chase.json'
                )
            with open(target_options_path, 'x') as f:
                json.dump(c, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
        elif opts.schedule_switch_manifest is not None:
            if not os.path.isfile(options_path):
                with open(options_path, 'x') as f:
                    json.dump(c, f, indent=2)
                    f.write('\n')
                    f.flush()
                    os.fsync(f.fileno())
        elif strict_factorial_resume:
            if not os.path.isfile(options_path):
                raise click.ClickException(
                    'strict factorial resume requires the immutable original '
                    'training_options.json in the same run directory'
                )
        else:
            mode = 'xt' if (
                opts.factorial_protocol in STRICT_FACTORIAL_PROTOCOLS
            ) else 'wt'
            with open(options_path, mode) as f:
                json.dump(c, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
