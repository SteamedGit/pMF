"""Benchmark compiled one-step pMF sampling with random weights (FP32)."""

import argparse
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from models import pmfDiT


def main():
    configs = {name.removeprefix('pmfDiT_'): factory
               for name, factory in vars(pmfDiT).items()
               if name.startswith('pmfDiT_') and callable(factory)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=configs, default='B_2')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Default: selected model configuration')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--omega', type=float, default=8.5)
    parser.add_argument('--t_min', type=float, default=0.1)
    parser.add_argument('--t_max', type=float, default=0.7)
    parser.add_argument('--noise_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', type=Path,
                        help='Also save settings and timings to a text file (overwritten)')
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 1 or args.runs < 2:
        parser.error('Require batch_size >= 1, warmup >= 1, and runs >= 2')
    if args.omega <= 0 or args.noise_scale <= 0:
        parser.error('omega and noise_scale must be positive')
    if not 0 <= args.t_min <= args.t_max <= 1:
        parser.error('Require 0 <= t_min <= t_max <= 1')

    overrides = {} if args.img_size is None else {'input_size': args.img_size}
    model = configs[args.model](eval=True, **overrides)
    args.img_size = model.input_size
    if args.img_size <= 0 or args.img_size % model.patch_size:
        parser.error('img_size must be positive and divisible by the model patch size')

    shape = (args.batch_size, args.img_size, args.img_size, 3)
    key, init_key, label_key = jax.random.split(jax.random.key(args.seed), 3)
    labels = jax.random.randint(label_key, (args.batch_size,), 0, model.num_classes)
    ones = jnp.ones((args.batch_size,), dtype=jnp.float32)
    conditioning = (ones, ones, ones * args.omega,
                    ones * args.t_min, ones * args.t_max, labels)
    variables = jax.jit(model.init)(
        init_key, jnp.zeros(shape, dtype=jnp.float32), *conditioning)
    jax.block_until_ready(variables)

    @jax.jit
    def sample(variables, key):
        noise = jax.random.normal(key, shape, dtype=jnp.float32) * args.noise_scale
        u, _ = model.apply(variables, noise, *conditioning)
        return noise - u  # One flow-map step: t=1, r=0, h=1.

    report = [
        'pMF one-step sampling benchmark', f'Device: {jax.devices()[0]}',
        'Random weights; steps: 1; precision: fp32; compile: True',
        'Timing includes noise generation and sampling; excludes initialization, '
        'compilation, warm-ups, and file saving.',
        '', 'Settings:', *(f'{k}: {v}' for k, v in vars(args).items()), '',
    ]
    print('\n'.join(report), flush=True)
    for i in range(args.warmup):
        key, sample_key = jax.random.split(key)
        sample(variables, sample_key).block_until_ready()
        print(f'Warm-up {i + 1}/{args.warmup} complete', flush=True)

    timings = []
    for i in range(args.runs):
        key, sample_key = jax.random.split(key)
        sample_key.block_until_ready()
        start = time.perf_counter()
        sample(variables, sample_key).block_until_ready()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        line = f'Run {i + 1}/{args.runs}: {elapsed:.6f} s'
        report.append(line)
        print(line, flush=True)

    mean = statistics.mean(timings)
    summary = [f'Mean: {mean:.6f} s/batch',
               f'Std: {statistics.stdev(timings):.6f} s/batch (sample standard deviation)',
               f'Throughput: {args.batch_size / mean:.2f} images/s']
    print('\n'.join(summary))
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text('\n'.join(report + [''] + summary) + '\n', encoding='utf-8')
        print(f'Saved results to {args.output_file}')


if __name__ == '__main__':
    main()
