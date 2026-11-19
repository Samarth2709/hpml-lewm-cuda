"""Latency benchmark, closed-loop PushT evaluation and statistics for the LeWM planners.

  python experiments.py benchmark --output results/<run>/benchmark.json
  python experiments.py evaluate --set eval96 --output results/<run>/eval96.json

A request = CPU current/goal images -> encode both -> 30 CEM rounds x 300 candidates x 5 steps
-> CPU plan. Cold start (first call) is recorded separately from warm repetitions, and the
configurations are interleaved in a shuffled order on every repetition.
"""
import os
os.environ.setdefault('TORCHINDUCTOR_USE_FAST_MATH', '0')  # strict FP32: no fast-math

import argparse
import datetime
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import model as M
import optimizations as O
import upstream

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Statistics (frozen protocol of 2026-11-10)
# ---------------------------------------------------------------------------

def bootstrap_median_ci(samples, resamples=10000, seed=20261109, confidence=0.95):
    """Percentile bootstrap confidence interval for the median."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        raise ValueError('samples must be non-empty')
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[index], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, [alpha, 1.0 - alpha])
    return [float(low), float(high)]


def bootstrap_speedup_ci(reference_samples, variant_samples, resamples=10000, seed=20261109, confidence=0.95):
    """Paired percentile bootstrap interval for median(reference) / median(variant)."""
    reference = np.asarray(reference_samples, dtype=np.float64)
    variant = np.asarray(variant_samples, dtype=np.float64)
    if reference.size == 0 or reference.size != variant.size:
        raise ValueError('reference and variant samples must have the same non-zero length')
    rng = np.random.default_rng(seed)
    index = rng.integers(0, reference.size, size=(resamples, reference.size))
    ratios = np.median(reference[index], axis=1) / np.median(variant[index], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(ratios, [alpha, 1.0 - alpha])
    ratio = np.median(reference) / np.median(variant)
    return {
        'ratio': float(ratio),
        'ci': [float(low), float(high)],
        'confidence': confidence,
        'resamples': resamples,
    }


def summarize_timings(samples, resamples=10000, seed=20261109, confidence=0.95):
    """Summary statistics for latency samples in milliseconds."""
    values = np.asarray(samples, dtype=np.float64)
    q25, q50, q75, p95 = np.quantile(values, [0.25, 0.50, 0.75, 0.95])
    return {
        'n': int(values.size),
        'median_ms': float(q50),
        'median_ci_ms': bootstrap_median_ci(values, resamples=resamples, seed=seed, confidence=confidence),
        'p95_ms': float(p95),
        'iqr_ms': float(q75 - q25),
        'mean_ms': float(values.mean()),
        'min_ms': float(values.min()),
        'max_ms': float(values.max()),
    }


def mcnemar_exact_p(losses, gains):
    """Exact two-sided McNemar p-value (binomial test on the discordant pairs)."""
    total = int(losses) + int(gains)
    if total == 0:
        return 1.0
    smaller = min(int(losses), int(gains))
    tail = 0
    for count in range(smaller + 1):
        tail += math.comb(total, count)
    tail = tail / (2 ** total)
    return float(min(1.0, 2.0 * tail))


def cluster_bootstrap_difference_ci(reference_successes, variant_successes, cluster_ids,
                                    resamples=10000, seed=20261110, confidence=0.95):
    """Two-sided cluster-bootstrap interval for (variant - reference) success rate.

    Episodes are resampled with replacement, so all seeds of one episode stay together.
    """
    reference = np.asarray(reference_successes, dtype=np.float64)
    variant = np.asarray(variant_successes, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    if reference.size == 0 or not (reference.size == variant.size == clusters.size):
        raise ValueError('Mismatched or empty inputs for cluster bootstrap')
    unique_clusters = np.unique(clusters)
    # Per-cluster sums of differences and counts make each resample a simple sum.
    differences = []
    counts = []
    for cluster in unique_clusters:
        in_cluster = clusters == cluster
        differences.append(float((variant[in_cluster] - reference[in_cluster]).sum()))
        counts.append(float(in_cluster.sum()))
    differences = np.array(differences)
    counts = np.array(counts)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, unique_clusters.size, size=(resamples, unique_clusters.size))
    boot_differences = differences[index].sum(axis=1) / counts[index].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_differences, [alpha, 1.0 - alpha])
    return [float(low), float(high)]


def quality_gate(records, reference_policy, variant_policy, margin=0.075,
                 resamples=10000, seed=20261110, confidence=0.95):
    """Paired closed-loop quality gate: pass if the 95% interval lower bound is above -margin."""
    reference_map = {}
    variant_map = {}
    for record in records:
        key = (record['row'], record['seed'])
        if record['policy'] == reference_policy:
            reference_map[key] = bool(record['success'])
        if record['policy'] == variant_policy:
            variant_map[key] = bool(record['success'])
    keys = sorted(set(reference_map) & set(variant_map))
    if not keys:
        raise ValueError(f'No paired episodes found between {reference_policy!r} and {variant_policy!r}')
    reference = [reference_map[key] for key in keys]
    variant = [variant_map[key] for key in keys]
    clusters = [key[0] for key in keys]

    gains = 0
    losses = 0
    for base_ok, variant_ok in zip(reference, variant):
        if variant_ok and not base_ok:
            gains += 1
        if base_ok and not variant_ok:
            losses += 1
    cases = len(keys)
    reference_rate = sum(reference) / cases
    variant_rate = sum(variant) / cases
    interval = cluster_bootstrap_difference_ci(reference, variant, clusters, resamples=resamples,
                                               seed=seed, confidence=confidence)
    return {
        'reference': reference_policy,
        'variant': variant_policy,
        'paired_cases': cases,
        'clusters': len(set(clusters)),
        'reference_successes': int(sum(reference)),
        'variant_successes': int(sum(variant)),
        'reference_rate': float(reference_rate),
        'variant_rate': float(variant_rate),
        'difference': float(variant_rate - reference_rate),
        'ci_95': interval,
        'margin': float(margin),
        'passed': bool(interval[0] > -margin),
        'gained': int(gains),
        'lost': int(losses),
        'mcnemar_p': mcnemar_exact_p(losses, gains),
    }


# ---------------------------------------------------------------------------
# Planner configurations
# ---------------------------------------------------------------------------

#   baseline           - released rollout and native CEM, eager PyTorch
#   baseline_compiled  - unchanged baseline + torch.compile + one CUDA graph per CEM round (CUDA)
#   optimized          - optimized rollout (optimizations.rollout), same search, eager PyTorch
#   optimized_tree     - optimized rollout on shared-prefix tree proposals ('late'), eager PyTorch
#   optimized_cuda               - same search, compiled rounds, 30-round CUDA graph (CUDA)
#   optimized_tree_noshared_cuda - optimized_tree_cuda without the shared sliding window (ablation)
#   optimized_tree_cuda          - tree search, compiled rounds, 30-round CUDA graph (CUDA)
#   *_triton                     - the same CUDA graph planner with the Triton short-attention kernel
CONFIGS = ['baseline', 'baseline_compiled', 'optimized', 'optimized_tree',
           'optimized_cuda', 'optimized_cuda_triton', 'optimized_tree_noshared_cuda',
           'optimized_tree_cuda', 'optimized_tree_cuda_triton']
CPU_CONFIGS = ['baseline', 'optimized', 'optimized_tree']
# Same-search configurations must reproduce the baseline plan up to FP32 rounding.
SAME_SEARCH = ['baseline_compiled', 'optimized', 'optimized_cuda', 'optimized_cuda_triton']
# Tree configurations change the proposals; they are compared with the same tree search
# evaluated leaf by leaf (the 'independent' rollout on tree-shaped candidates).
TREE_SEARCH = ['optimized_tree', 'optimized_tree_noshared_cuda', 'optimized_tree_cuda',
               'optimized_tree_cuda_triton']
# Whole-search CUDA graph planners can serve R requests together (multi-request batching).
GRAPH_CONFIGS = ['optimized_cuda', 'optimized_cuda_triton', 'optimized_tree_noshared_cuda',
                 'optimized_tree_cuda', 'optimized_tree_cuda_triton']

PLANNERS = {}


def cuda_planner(model, name, batch_size=1):
    """Build each CUDA planner once and reuse it; compilation and capture are cold-start cost."""
    key = (id(model), name, batch_size)
    if key in PLANNERS:
        return PLANNERS[key]
    device = str(next(model.parameters()).device)
    if not device.startswith('cuda'):
        raise ValueError(f'{name} requires --device cuda')
    if name == 'baseline_compiled':
        if batch_size != 1:
            raise ValueError('baseline_compiled serves one request at a time')
        planner = O.CompiledGraphBaseline(model)
    elif name == 'optimized_cuda':
        planner = O.WholeSearchCUDAGraph(model, 'independent', batch_size=batch_size)
    elif name == 'optimized_cuda_triton':
        planner = O.WholeSearchCUDAGraph(model, 'independent', use_cuda_attn=True, batch_size=batch_size)
    elif name == 'optimized_tree_noshared_cuda':
        planner = O.WholeSearchCUDAGraph(model, 'late', batch_size=batch_size, use_shared_window=False)
    elif name == 'optimized_tree_cuda':
        planner = O.WholeSearchCUDAGraph(model, 'late', batch_size=batch_size)
    elif name == 'optimized_tree_cuda_triton':
        planner = O.WholeSearchCUDAGraph(model, 'late', use_cuda_attn=True, batch_size=batch_size)
    else:
        raise ValueError(f'Unknown configuration {name!r}')
    PLANNERS[key] = planner
    return planner


def tree_rollout(model, z, actions):
    return O.rollout(model, z, actions, layout='late')


def run(model, pair, seed, name):
    """Plan one request with configuration `name`. Returns the CPU plan dictionary."""
    pixels, goal = pair
    if name == 'baseline':
        return M.request(model, pixels, goal, seed)
    if name == 'optimized':
        return M.request(model, pixels, goal, seed, rollout=O.rollout)
    if name == 'optimized_tree':
        plan = O.tree_plan('late', str(next(model.parameters()).device))
        return M.request(model, pixels, goal, seed, rollout=tree_rollout, plan=plan)
    planner = cuda_planner(model, name)
    return planner.request(pixels, goal, seed)


def max_difference(first, second):
    result = {}
    for key in first:
        result[key] = float((first[key] - second[key]).abs().max())
    return result


def plans_equal(first, second):
    for key in first:
        if not torch.equal(first[key], second[key]):
            return False
    return True


def output_checks(model, pair, seed, outputs):
    """Largest absolute plan differences against the matching reference search."""
    checks = {}
    if 'baseline' in outputs:
        for name in SAME_SEARCH:
            if name in outputs:
                checks[f'{name}_vs_baseline_max_abs'] = max_difference(outputs['baseline'], outputs[name])
    tree_names = [name for name in TREE_SEARCH if name in outputs]
    if tree_names:
        pixels, goal = pair
        plan = O.tree_plan('late', str(next(model.parameters()).device))
        reference = M.request(model, pixels, goal, seed, rollout=O.rollout, plan=plan)
        for name in tree_names:
            if name == 'optimized_tree':
                checks['tree_shared_equals_independent'] = plans_equal(reference, outputs[name])
                checks['tree_shared_vs_independent_max_abs'] = max_difference(reference, outputs[name])
            else:
                checks[f'{name}_vs_tree_independent_max_abs'] = max_difference(reference, outputs[name])
    return checks


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def gpu_snapshot():
    if shutil.which('nvidia-smi') is None:
        return None
    fields = 'name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu'
    output = run_command(['nvidia-smi', f'--query-gpu={fields}', '--format=csv'])
    lines = None
    if output:
        lines = output.splitlines()
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'csv': lines}


def environment_record(device):
    status = run_command(['git', '-C', str(ROOT), 'status', '--porcelain'])
    record = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'device': device,
        'torch': torch.__version__,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'git_commit': run_command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        'git_dirty': bool(status),
        'strict_fp32': {
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        },
    }
    if device.startswith('cuda') and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        record['cuda'] = {
            'version': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'name': properties.name,
            'capability': list(torch.cuda.get_device_capability(device)),
            'sm_count': properties.multi_processor_count,
            'total_memory_bytes': properties.total_memory,
        }
        record['gpu_before'] = gpu_snapshot()
    return record


def save_report(report, output):
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


def print_scene_table(scene, args, configs, summary, speedups_vs_baseline, speedups_vs_compiled):
    print(f'\nScene {scene}, seed {args.seed} ({args.device}, torch {torch.__version__}, n={args.repeats})')
    print('| Config | Median ms (95% CI) | p95 ms | IQR ms | vs baseline | vs compiled |')
    print('|---|---:|---:|---:|---:|---:|')
    for name in configs:
        stats = summary[name]
        low, high = stats['median_ci_ms']
        versus_baseline = '-'
        if name in speedups_vs_baseline:
            versus_baseline = f"{speedups_vs_baseline[name]['ratio']:.2f}x"
        versus_compiled = '-'
        if name in speedups_vs_compiled:
            versus_compiled = f"{speedups_vs_compiled[name]['ratio']:.2f}x"
        print(f"| {name} | {stats['median_ms']:.1f} [{low:.1f}, {high:.1f}] | {stats['p95_ms']:.1f} "
              f"| {stats['iqr_ms']:.1f} | {versus_baseline} | {versus_compiled} |")


def measure_batch_sweep(model, args, configs):
    """Throughput of R requests served together by one CUDA graph planner.

    Request i of a batch uses development scene (scene + i) mod 24 and seed (seed + i).
    """
    graph_configs = [name for name in configs if name in GRAPH_CONFIGS]
    if not args.device.startswith('cuda') or not graph_configs:
        return {'skipped': True, 'reason': 'Requires CUDA and at least one whole-search CUDA graph config'}
    scene = args.scenes[0]
    results = {}
    for name in graph_configs:
        results[name] = {}
        for batch_size in args.batch_sizes:
            pairs = [M.image_pair((scene + index) % 24) for index in range(batch_size)]
            pixels = torch.cat([pair[0] for pair in pairs], dim=0)
            goals = torch.cat([pair[1] for pair in pairs], dim=0)
            seeds = [args.seed + index for index in range(batch_size)]
            planner = cuda_planner(model, name, batch_size)

            sync(args.device)
            start = time.perf_counter()
            output = planner.request(pixels, goals, seeds)
            sync(args.device)
            cold_ms = (time.perf_counter() - start) * 1e3
            # The batched plan of request 0 must match the single-request plan.
            single = run(model, pairs[0], seeds[0], name)
            difference = float((output['actions'][0:1] - single['actions']).abs().max())

            for _ in range(args.warmups):
                planner.request(pixels, goals, seeds)
            samples = []
            for _ in range(args.batch_repeats):
                sync(args.device)
                start = time.perf_counter_ns()
                planner.request(pixels, goals, seeds)
                sync(args.device)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            stats = summarize_timings(samples, resamples=args.bootstrap_resamples)
            stats['cold_start_ms'] = cold_ms
            stats['throughput_req_s'] = batch_size / (stats['median_ms'] / 1e3)
            stats['ms_per_request'] = stats['median_ms'] / batch_size
            stats['req0_max_abs_diff'] = difference
            results[name][str(batch_size)] = stats
            print(f"[batch] {name} R={batch_size}: median={stats['median_ms']:.1f} ms "
                  f"({stats['ms_per_request']:.1f} ms/req, {stats['throughput_req_s']:.2f} req/s, "
                  f"diff={difference:.2e})")
    return results


def benchmark(args):
    configs = args.configs
    if configs is None and args.device.startswith('cuda'):
        configs = CONFIGS
    if configs is None:
        configs = CPU_CONFIGS
    model = M.load_model(args.device)
    report = {
        'environment': environment_record(args.device),
        'configs': configs,
        'seed': args.seed,
        'repeats': args.repeats,
        'warmups': args.warmups,
        'scenes': {},
    }

    for scene in args.scenes:
        pair = M.image_pair(scene)
        cold_start_ms = {}
        outputs = {}
        for name in configs:
            sync(args.device)
            start = time.perf_counter()
            outputs[name] = run(model, pair, args.seed, name)
            sync(args.device)
            cold_start_ms[name] = (time.perf_counter() - start) * 1e3
        checks = output_checks(model, pair, args.seed, outputs)

        for name in configs:
            for _ in range(args.warmups):
                run(model, pair, args.seed, name)

        samples = {}
        for name in configs:
            samples[name] = []
        orders = []
        rng = random.Random(2027 + scene)
        for _ in range(args.repeats):
            order = rng.sample(configs, len(configs))
            orders.append(order)
            for name in order:
                sync(args.device)
                start = time.perf_counter_ns()
                run(model, pair, args.seed, name)
                sync(args.device)
                samples[name].append((time.perf_counter_ns() - start) / 1e6)

        summary = {}
        for name in configs:
            summary[name] = summarize_timings(samples[name], resamples=args.bootstrap_resamples)
        speedups_vs_baseline = {}
        if 'baseline' in samples:
            for name in configs:
                speedups_vs_baseline[name] = bootstrap_speedup_ci(
                    samples['baseline'], samples[name], resamples=args.bootstrap_resamples)
        speedups_vs_compiled = {}
        if 'baseline_compiled' in samples:
            for name in configs:
                speedups_vs_compiled[name] = bootstrap_speedup_ci(
                    samples['baseline_compiled'], samples[name], resamples=args.bootstrap_resamples)

        report['scenes'][str(scene)] = {
            'checks': checks,
            'cold_start_ms': cold_start_ms,
            'samples_ms': samples,
            'orders': orders,
            'summary': summary,
            'speedups_vs_baseline': speedups_vs_baseline,
            'speedups_vs_baseline_compiled': speedups_vs_compiled,
        }
        print_scene_table(scene, args, configs, summary, speedups_vs_baseline, speedups_vs_compiled)
        rounded = {name: round(value, 1) for name, value in cold_start_ms.items()}
        print('Cold start (ms):', rounded)
        print('Checks:', json.dumps(checks))
        save_report(report, args.output)

    if args.batch_sizes:
        print('\nRunning multi-request throughput sweep...')
        batch_configs = args.batch_configs or configs
        report['batch_sweep'] = measure_batch_sweep(model, args, batch_configs)
        save_report(report, args.output)

    if args.device.startswith('cuda'):
        report['environment']['gpu_after'] = gpu_snapshot()
    save_report(report, args.output)


# ---------------------------------------------------------------------------
# Closed-loop PushT evaluation
# ---------------------------------------------------------------------------
# Reset to a recorded start (simulator seed 42), goal = recorded state 25 commands later,
# plan 25 commands, execute, replan once, execute (50 commands). Success = the simulator's
# success region is reached at any command.

#   dev24  - 24 development episodes x 2 seeds = 48 paired cases
#   eval96 - 96 frozen evaluation episodes x 3 seeds = 288 paired cases (quality gate)
SETS = {
    'dev24': ('development_scenes.npz', 24, (42, 43)),
    'eval96': ('pusht_96_episode_evaluation_cohort.npz', 96, (42, 43, 44)),
}


def check_step(observation, truncated):
    if truncated or not np.isfinite(observation['state']).all():
        raise RuntimeError('Simulator truncated the episode or returned a non-finite state')


def episode(data, row, policy, model, seed, scale, offset):
    simulator = upstream.create_simulator()  # fresh simulator per episode, as in the original protocol
    try:
        start = data['states'][row, 0]
        goal = data['states'][row, 25]
        observation, info = simulator.reset(start, goal, 42)
        goal_image = np.asarray(info['goal']).copy()
        hits = []
        commands = 0
        for replan in range(2):
            pixels = upstream.preprocess(observation['pixels'][None]).unsqueeze(1)
            goal_pixels = upstream.preprocess(goal_image[None]).unsqueeze(1)
            plan = run(model, (pixels, goal_pixels), seed + replan, policy)
            raw = plan['actions'][0].reshape(25, 2).numpy().copy()
            # Inverse action normalization, in place in FP32 exactly as the original protocol.
            raw *= scale
            raw += offset
            for action in raw:
                observation, _, success, truncated, _ = simulator.step(action)
                check_step(observation, truncated)
                commands += 1
                if success:
                    hits.append(commands)
        final = np.asarray(observation['state']).copy()
    finally:
        simulator.close()

    angle = abs(float(final[4] - goal[4]))
    first_success = None
    if hits:
        first_success = hits[0]
    return {
        'row': row,
        'seed': seed,
        'policy': policy,
        'success': bool(hits),
        'first_success': first_success,
        'position_error': float(np.linalg.norm(final[:4] - goal[:4])),
        'angle_error': min(angle, 2 * np.pi - angle),
    }


def run_control(data, row, kind):
    """Sanity controls: 'expert_25' (recorded 25 commands) or 'zero_50' (50 zero commands)."""
    simulator = upstream.create_simulator()
    try:
        start = data['states'][row, 0]
        goal = data['states'][row, 25]
        simulator.reset(start, goal, 42)
        if kind == 'expert_25':
            actions = data['raw_actions'][row, :25]
        else:
            actions = np.zeros((50, 2), dtype=np.float32)
        hits = []
        for step, action in enumerate(actions, start=1):
            observation, _, success, truncated, _ = simulator.step(action)
            check_step(observation, truncated)
            if success:
                hits.append(step)
    finally:
        simulator.close()
    first_success = None
    if hits:
        first_success = hits[0]
    return {'row': row, 'control': kind, 'success': bool(hits), 'first_success': first_success}


def evaluate(args):
    file_name, episode_count, seeds = SETS[args.set]
    rows = list(range(episode_count))
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    with np.load(M.asset(file_name)) as archive:
        data = {'states': archive['states'], 'raw_actions': archive['raw_actions']}
    normalization = M.config_section('action_normalization')
    scale = np.array(normalization['scale'])
    offset = np.array(normalization['mean'])

    controls = {}
    if args.controls:
        for kind in ('expert_25', 'zero_50'):
            results = [run_control(data, row, kind) for row in rows]
            successes = sum(result['success'] for result in results)
            controls[kind] = f'{successes}/{len(results)}'
            print(f'Control {kind}: {controls[kind]}', flush=True)

    model = M.load_model(args.device)
    environment = environment_record(args.device)
    records = []
    for row in rows:
        for seed in seeds:
            for policy in args.policies:
                records.append(episode(data, row, policy, model, seed, scale, offset))
                print(json.dumps(records[-1]), flush=True)

    reference_policy = args.policies[0]
    summary = {}
    gates = {}
    for policy in args.policies:
        policy_records = [record for record in records if record['policy'] == policy]
        gate = quality_gate(records, reference_policy, policy, margin=args.margin,
                            resamples=args.bootstrap_resamples)
        gates[policy] = gate
        summary[policy] = {
            'successes': f"{gate['variant_successes']}/{gate['paired_cases']}",
            'rate': round(gate['variant_rate'], 4),
            'diff_vs_ref': round(gate['difference'], 4),
            'ci_95': [round(value, 4) for value in gate['ci_95']],
            'gate_passed': gate['passed'],
            'gained': gate['gained'],
            'lost': gate['lost'],
            'mcnemar_p': round(gate['mcnemar_p'], 4),
            'median_position_error': statistics.median(record['position_error'] for record in policy_records),
            'median_angle_error': statistics.median(record['angle_error'] for record in policy_records),
        }
    print(json.dumps(summary, indent=2))
    report = {
        'set': args.set,
        'episodes': len(rows),
        'seeds': list(seeds),
        'device': args.device,
        'torch': torch.__version__,
        'environment': environment,
        'reference_policy': reference_policy,
        'margin': args.margin,
        'controls': controls,
        'summary': summary,
        'gates': gates,
        'records': records,
    }
    save_report(report, args.output)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench = commands.add_parser('benchmark', help='Full-request latency of each configuration.')
    bench.add_argument('--device', default=default_device)
    bench.add_argument('--configs', nargs='+', choices=CONFIGS, default=None)
    bench.add_argument('--scenes', type=int, nargs='+', default=[12])
    bench.add_argument('--seed', type=int, default=47)
    bench.add_argument('--repeats', type=int, default=30)
    bench.add_argument('--warmups', type=int, default=3)
    bench.add_argument('--bootstrap-resamples', type=int, default=10000)
    bench.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                       help='Optional multi-request throughput sweep, for example 1 2 4 8.')
    bench.add_argument('--batch-configs', nargs='+', choices=GRAPH_CONFIGS, default=None,
                       help='Configurations for the sweep (default: the CUDA graph configs being benchmarked).')
    bench.add_argument('--batch-repeats', type=int, default=20)
    bench.add_argument('--output', type=Path)

    quality = commands.add_parser('evaluate', help='Closed-loop PushT task quality of each planner.')
    quality.add_argument('--device', default=default_device)
    quality.add_argument('--set', choices=SETS, default='eval96')
    quality.add_argument('--policies', nargs='+', choices=CONFIGS, default=['baseline'])
    quality.add_argument('--max-rows', type=int, default=None, help='Optional cap on episodes (smoke tests).')
    quality.add_argument('--controls', action=argparse.BooleanOptionalAction, default=True,
                         help='Run the expert_25 and zero_50 simulator controls (default: on).')
    quality.add_argument('--margin', type=float, default=0.075)
    quality.add_argument('--bootstrap-resamples', type=int, default=10000)
    quality.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()
    if args.device.startswith('cuda'):
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(2)
    if args.command == 'benchmark':
        benchmark(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
