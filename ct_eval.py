import os
import re
import json
import click

import pickle
import psutil
import functools
import PIL.Image
import numpy as np

import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import metric_main, metric_utils

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

IMAGENET64_FEATURE_COUNT = 50_000
IMAGENET64_FEATURE_DIM = 2_048
IMAGENET64_CLASS_COUNT = 1_000
IMAGENET64_METRIC_SEED = 20_260_730
IMAGENET64_NFE2_MID_T = 1.526
INCEPTION_DETECTOR_URL = metric_utils.OFFICIAL_EDM2_INCEPTION_URL

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

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm|edm2',     type=click.Choice(['ddpmpp', 'ncsnpp', 'adm', 'edm2']), default='ddpmpp', show_default=True)
@click.option('--preset',        help='Model configuration preset', metavar='STR',                  type=click.Choice(['edm2-img64-s']), default=None)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|ct',    type=click.Choice(['vp', 've', 'edm', 'ct']), default='ct', show_default=True)

# Hyperparameters.
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--dropres',       help='Feature resolution where EDM2 dropout is applied', metavar='INT', type=click.IntRange(min=1))
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)

@click.option('--scheduler',     help='Type of consistency scheduler', metavar='STR',               type=click.Choice(['logsnr', 'power', 'sigmoid']), default='sigmoid', show_default=True)
@click.option('--double',        help='How often to save latest checkpoints', metavar='TICKS',      type=click.IntRange(min=1), default=500, show_default=True)

@click.option('-q',              help='Decay Factor', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1.4, show_default=True)
@click.option('-c',              help='Constant c for Huber Loss', metavar='FLOAT',                 type=click.FloatRange(), default=0.0, show_default=True)
@click.option('-k',              help='Consistency condition hyperparams.', metavar='FLOAT',        type=click.FloatRange(), default=8.0, show_default=True)
@click.option('-b',              help='Consistency condition hyperparams.', metavar='FLOAT',        type=click.FloatRange(), default=1.0, show_default=True)
@click.option('--cut',           help='Cutoff value.', metavar='FLOAT',                             type=click.FloatRange(), default=4.0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--eval-batch',    help='Batch size for evaluator previews/module summary', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--metric-generator-batch', help='Generator microbatch used by metrics', metavar='INT', type=click.IntRange(min=1), default=128, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Load network pickle', metavar='PKL|URL',   type=str)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.821]',                             multiple=True, default=[0.821])
@click.option('--nfe',           help='Number of function evaluations',                            type=click.Choice(['1', '2']), default='2', show_default=True)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--metric-repeats', help='Number of times to repeat each metric',                     type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--sample-seeds',  help='Explicit per-sample seed list/range (single-GPU only)',       metavar='LIST', type=str)
@click.option('--retain-generated-artifacts', help='Retain exact generated samples and metric features', is_flag=True)
@click.option('--feature-only', help='Extract formal ImageNet-64 Inception features without images or quality metrics', is_flag=True)
@click.option('--feature-output', help='Output .npy path for --feature-only', metavar='NPY', type=str)
@click.option('--engineering-feature-count', help='Engineering-only feature extraction count below 50000', metavar='INT', type=click.IntRange(min=2, max=IMAGENET64_FEATURE_COUNT - 1))


def validate_imagenet64_feature_contract(
    opts, sample_seeds, world_size, resolution, num_channels, label_dim,
):
    if opts.arch != 'edm2' or opts.preset != 'edm2-img64-s':
        raise click.ClickException(
            '--feature-only requires --arch=edm2 --preset=edm2-img64-s'
        )
    if not opts.resume:
        raise click.ClickException('--feature-only requires --resume=CHECKPOINT')
    if (
        resolution != 64
        or num_channels != 3
        or not opts.cond
        or label_dim != IMAGENET64_CLASS_COUNT
    ):
        raise click.ClickException(
            '--feature-only requires a conditional ImageNet-64 dataset with '
            f'{IMAGENET64_CLASS_COUNT} classes'
        )
    if opts.fp16:
        raise click.ClickException('--feature-only requires --fp16=False')
    if world_size != 1:
        raise click.ClickException('--feature-only requires exactly one GPU')
    feature_count = getattr(opts, 'engineering_feature_count', None)
    feature_count = IMAGENET64_FEATURE_COUNT if feature_count is None else feature_count
    if sample_seeds != list(range(feature_count)):
        raise click.ClickException(
            f'--feature-only requires --sample-seeds=0-{feature_count - 1}'
        )
    if opts.metrics:
        raise click.ClickException('--feature-only requires --metrics=none')
    if opts.retain_generated_artifacts:
        raise click.ClickException(
            '--feature-only does not permit retained generated samples'
        )
    if opts.seed is not None and opts.seed != IMAGENET64_METRIC_SEED:
        raise click.ClickException(
            f'--feature-only requires --seed={IMAGENET64_METRIC_SEED}'
        )
    if opts.nfe == '2' and tuple(opts.mid_t) != (IMAGENET64_NFE2_MID_T,):
        raise click.ClickException(
            f'ImageNet-64 NFE=2 requires --mid_t={IMAGENET64_NFE2_MID_T}'
        )


def main(**kwargs):
    """Train ECMs using the techniques described in the 
    blog "Consistency Models Made Easy".
    """   
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if opts.preset == 'edm2-img64-s':
        if opts.arch != 'edm2':
            raise click.ClickException('edm2-img64-s requires --arch=edm2')
        opts.cbase = 192
        opts.dropout = 0.40
        opts.dropres = 16
        opts.augment = 0
    if opts.feature_output is not None and not opts.feature_only:
        raise click.ClickException('--feature-output requires --feature-only')

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.network_kwargs = dnnlib.EasyDict()

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        dataset_label_dim = dataset_obj.label_dim
        dataset_num_channels = dataset_obj.num_channels
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
    elif opts.arch == 'adm':
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
    else:
        assert opts.arch == 'edm2'
        if opts.preset != 'edm2-img64-s':
            raise click.ClickException('--arch=edm2 requires --preset=edm2-img64-s')
        c.network_kwargs.update(
            class_name='training.networks_edm2.Precond',
            model_channels=192,
            dropout=0.40,
            dropout_res=16,
        )

    if opts.arch != 'edm2':
        c.network_kwargs.class_name = 'training.networks.ECMPrecond'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Trainig options.
    c.update(cudnn_benchmark=opts.bench)
    sample_seeds = None if opts.sample_seeds is None else parse_int_list(opts.sample_seeds)
    if sample_seeds is not None:
        if len(sample_seeds) == 0:
            raise click.ClickException('--sample-seeds must not be empty')
        if len(set(sample_seeds)) != len(sample_seeds):
            raise click.ClickException('--sample-seeds must not contain duplicates')
        if dist.get_world_size() != 1:
            raise click.ClickException('--sample-seeds currently requires exactly one GPU')
    if opts.retain_generated_artifacts and dist.get_world_size() != 1:
        raise click.ClickException('--retain-generated-artifacts currently requires exactly one GPU')
    if 128 % opts.metric_generator_batch != 0:
        raise click.ClickException('--metric-generator-batch must divide the metric batch size 128')
    if opts.feature_only:
        validate_imagenet64_feature_contract(
            opts,
            sample_seeds,
            dist.get_world_size(),
            c.dataset_kwargs.resolution,
            dataset_num_channels,
            dataset_label_dim,
        )
    c.update(
        batch_size=opts.eval_batch,
        mid_t=() if opts.nfe == '1' else opts.mid_t,
        metrics=opts.metrics,
        metric_repeats=opts.metric_repeats,
        sample_seeds=sample_seeds,
        retain_generated_artifacts=opts.retain_generated_artifacts,
        metric_generator_batch=opts.metric_generator_batch,
        feature_only=opts.feature_only,
        feature_output=opts.feature_output,
        feature_count=(
            opts.engineering_feature_count
            if opts.engineering_feature_count is not None
            else IMAGENET64_FEATURE_COUNT
        ),
        balanced_class_labels=(IMAGENET64_CLASS_COUNT if opts.feature_only else None),
    )

    # Random seed.
    if opts.feature_only:
        c.seed = IMAGENET64_METRIC_SEED
    elif opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Checkpoint to evaluate.
    c.resume_pkl = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-{dtype_str:s}'
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
    dist.print0('Evaluarion options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
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
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    evaluation(**c)

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

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, 
    t_max=80, mid_t=None, step_noises=None, sample_seeds=None,
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    round_sigma = getattr(net, 'round_sigma', None)
    t_steps = torch.cat([
        round_sigma(t_steps) if round_sigma is not None else t_steps,
        torch.zeros_like(t_steps[:1]),
    ])

    intermediate_steps = max(len(t_steps) - 2, 0)
    if step_noises is not None and len(step_noises) != intermediate_steps:
        raise ValueError('step_noises must contain one tensor per intermediate sampling step')
    if sample_seeds is not None:
        if step_noises is not None:
            raise ValueError('sample_seeds and step_noises are mutually exclusive')
        if len(sample_seeds) != latents.shape[0]:
            raise ValueError('sample_seeds must contain one seed per latent')
        seeded_noises = [[] for _ in range(intermediate_steps)]
        shape = tuple(latents.shape[1:])
        for seed in sample_seeds:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            # Consume the matching latent draw before deriving step noise. This
            # mirrors scripts/sample_fixed_seeds.py and keeps NFE=1/2 paired.
            torch.randn(shape, generator=generator, dtype=torch.float64)
            for index in range(intermediate_steps):
                seeded_noises[index].append(
                    torch.randn(shape, generator=generator, dtype=torch.float64)
                )
        step_noises = [torch.stack(items).to(latents.device) for items in seeded_noises]

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            noise = torch.randn_like(x) if step_noises is None else step_noises[i]
            if noise.shape != x.shape:
                raise ValueError(f'step_noises[{i}] has shape {noise.shape}, expected {x.shape}')
            x = x + t_next * noise.to(device=x.device, dtype=x.dtype)
    return x

#----------------------------------------------------------------------------

def evaluation(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    network_kwargs      = {},       # Options for model and preconditioning.
    batch_size          = 512,      # Total batch size for one training iteration.
    seed                = 0,        # Global random seed.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    metric_repeats      = 3,        # Number of deterministic repeats per metric.
    sample_seeds        = None,     # Explicit per-sample seeds for proxy metrics.
    retain_generated_artifacts = False, # Save exact generated samples/features used by metrics.
    metric_generator_batch = 128, # Generator microbatch used inside metric feature extraction.
    feature_only        = False,    # Extract features without previews or metric values.
    feature_output      = None,     # Destination for generated features.
    feature_count       = None,     # Formal count, or explicit engineering gate count.
    balanced_class_labels = None,   # Direct one-hot label count for formal generation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.eval().requires_grad_(False).to(device)
    if dist.get_rank() == 0 and not feature_only:
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
        misc.copy_params_and_buffers(
            src_module=data['ema'], dst_module=net, require_all=feature_only
        )
        del data # conserve memory

    if feature_only:
        feature_count = IMAGENET64_FEATURE_COUNT if feature_count is None else feature_count
        output_path = feature_output or os.path.join(run_dir, 'generated-features.npy')
        if os.path.exists(output_path):
            raise FileExistsError(f'feature output already exists: {output_path}')
        if feature_count != IMAGENET64_FEATURE_COUNT:
            dist.print0(
                f'ENGINEERING GATE ONLY: extracting {feature_count} features; '
                'this is not a formal quality artifact.'
            )
        dist.print0('Extracting generated Inception features...')
        opts = metric_utils.MetricOptions(
            generator_fn=functools.partial(generator_fn, mid_t=mid_t),
            G=net,
            G_kwargs={},
            dataset_kwargs=dataset_kwargs,
            num_gpus=1,
            rank=0,
            device=device,
            sample_seeds=sample_seeds,
            metric_seed=seed,
            generator_batch_size=metric_generator_batch,
            balanced_class_labels=balanced_class_labels,
        )
        stats = metric_utils.compute_feature_stats_for_generator(
            opts,
            detector_url=INCEPTION_DETECTOR_URL,
            detector_kwargs={'return_features': True},
            batch_size=128,
            capture_all=True,
            max_items=feature_count,
        )
        features = stats.get_all()
        valid = (
            features.shape == (feature_count, IMAGENET64_FEATURE_DIM)
            and features.dtype == np.float32
        )
        for start in range(0, features.shape[0], 4096):
            valid = valid and np.isfinite(features[start:start + 4096]).all()
        if not valid:
            raise RuntimeError(
                'generated features must be finite float32 with shape '
                f'({feature_count}, {IMAGENET64_FEATURE_DIM})'
            )
        metric_utils._atomic_save_npy(output_path, features)
        dist.print0(
            'Feature extraction complete: '
            f'count={features.shape[0]} dimension={features.shape[1]} dtype=float32'
        )
        return
    
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], net.img_channels, net.img_resolution, net.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
        
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(net, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'sample.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for repeat_index in range(metric_repeats):
        shared_generated_features_path = None
        shared_generated_features_metric = None
        for metric_index, metric in enumerate(metrics):
            generated_features_path = None
            generated_samples_path = None
            if retain_generated_artifacts and dist.get_rank() == 0:
                generated_features_path = os.path.join(
                    run_dir, f'generated-features-{metric}-repeat{repeat_index:02d}.npy')
                if repeat_index == 0 and metric_index == 0:
                    generated_samples_path = os.path.join(run_dir, 'generated-samples.npy')
            result_dict = metric_main.calc_metric(metric=metric, 
                generator_fn=few_step_fn, G=net, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device,
                sample_seeds=sample_seeds, metric_seed=seed,
                generated_features_path=generated_features_path,
                generated_samples_path=generated_samples_path,
                generator_batch_size=metric_generator_batch,
                precomputed_generated_features_path=(
                    shared_generated_features_path
                    if (shared_generated_features_metric, metric)
                    == ('kid50k_full', 'fid50k_full')
                    else None
                ),
                precomputed_generated_features_source_metric=(
                    shared_generated_features_metric
                    if (shared_generated_features_metric, metric)
                    == ('kid50k_full', 'fid50k_full')
                    else None
                ))
            if (
                metric == 'kid50k_full'
                and generated_features_path is not None
                and shared_generated_features_path is None
            ):
                shared_generated_features_path = generated_features_path
                shared_generated_features_metric = metric
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'{resume_pkl}')

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
