# GPU Optimization for LeWorldModel Planning

**Preliminary results, 22 November 2026.**

## Project goal

Speed up LeWorldModel planning on NVIDIA GPUs while preserving task success.
We keep the pretrained weights, strict 32-bit floating-point precision (FP32),
and search budget fixed, and compare the original and optimized planners.

## Background context

- **Model:** [LeWorldModel](https://github.com/lucas-maes/le-wm), using its released
  [PushT checkpoint](https://huggingface.co/quentinll/lewm-pusht). A vision
  transformer encodes images, and a predictor estimates the outcomes of actions.
  Relevant paper: Maes et al.,
  [*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*](https://arxiv.org/abs/2603.19312).
- **Benchmark:** PushT, where a pusher moves a T-shaped block onto a target.
  We use the provided data, with 24 development scenes and 96 evaluation starts.
  The evaluation starts are separate from our tuning/development starts but come
  from the model's training-data source.
- **Hardware:** One NVIDIA L4 GPU on AWS EC2 `g6.xlarge`. The measurement settings
  are listed under current progress.

## Optimization / technical methodology

The planner uses the cross-entropy method (CEM): test 300 five-step action plans,
keep the best 30, and generate improved candidates for 30 rounds.
The implementation is in [optimizations.py](optimizations.py).

| Public commit date (2026) | Implementation milestone |
| --- | --- |
| [20 October](https://github.com/Samarth2709/hpml-lewm-cuda/commit/bea403f) | Establish the strict-FP32 baseline, model loading, benchmark and evaluation pipeline. |
| [23 October](https://github.com/Samarth2709/hpml-lewm-cuda/commit/cf82200) | Compile the original prediction/cost function and replay one CUDA graph per search round. |
| [24 October](https://github.com/Samarth2709/hpml-lewm-cuda/commit/13126fc) | Reuse earlier attention calculations during the first three predicted steps, while their positions remain valid. |
| [27 October](https://github.com/Samarth2709/hpml-lewm-cuda/commit/9a8af55) | Skip unused older outputs in the final predictor block. |
| [30 October](https://github.com/Samarth2709/hpml-lewm-cuda/commit/42c3628) | Compile small pointwise operations together to reduce separate GPU operations. |
| [2 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/b49d9a1) | Specialize attention for the one-to-three-state context. |
| [4 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/b159151) | Share candidate action prefixes, evaluating a shared beginning once. This changes the search proposals. |
| [7 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/eb16a40) | Share recent-history calculations between branches of the planning tree. |
| [10 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/4eda9d8) | Compile a complete round: candidate construction, prediction, scoring, elite selection and search updates. |
| [13 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/e3d1a1f) | Capture all 30 search rounds in one CUDA graph and replay them to reduce CPU launches. |
| [16 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/4785cba) | Try a custom Triton short-attention kernel. |
| [19 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/104913c) | Batch independent planning requests and add throughput measurements. |
| [22 November](https://github.com/Samarth2709/hpml-lewm-cuda/commit/81ed6ea) | Retain the baseline and final CUDA planners, include saved results, and remove superseded experimental paths, including custom Triton attention. |

`optimized_cuda` preserves the original search. `optimized_tree_cuda` shares
paths and changes the proposals, so its task success is checked separately.

## Profiling, benchmarking & evaluation tools

- **PyTorch Profiler:** Inspect CPU/CUDA work in encoding, prediction and search.
  Used during exploratory development; a matched profile of the strict-FP32
  planners remains pending. Profiler timings are separate from benchmark timings.
- **Timing:** Python `time.perf_counter_ns`, with `torch.cuda.synchronize` before
  and after each measurement. Configurations run in shuffled, interleaved order;
  cold compilation and graph capture are recorded separately.
- **Environment:** `nvidia-smi` and JSON records capture GPU/software settings.
- **Quality:** The PushT harness in [experiments.py](experiments.py) compares
  success/failure on paired starts and seeds, with expert-action and zero-action
  controls. NumPy computes median latency and 95% bootstrap intervals with
  10,000 resamples. Task-quality intervals resample whole episodes, keeping each
  episode's solver seeds together.

The saved results do not include a completed Nsight Systems capture.

## Project links

- **GitHub:** [Samarth2709/hpml-lewm-cuda](https://github.com/Samarth2709/hpml-lewm-cuda).
- **Experiment tracking:** [Public experiment record](https://github.com/Samarth2709/hpml-lewm-cuda#experiment-record).
  We use this record and its linked JSON artifacts for tracking.

## Current progress

### Baseline & experimental setup

`baseline` runs the released model rollout with eager PyTorch.
`baseline_compiled` keeps that rollout and search, adds `torch.compile`, and
replays one CUDA graph per round. Compilation uses
`mode='max-autotune-no-cudagraphs'`; graph capture is explicit. The compiled
baseline is the speedup reference below.

| Setting | Current FP32 experiments |
| --- | --- |
| Hardware | One NVIDIA L4, 24 GB nominal GPU memory, AWS `g6.xlarge` |
| Recorded GPU/runtime | 23,661,248,512 bytes GPU memory; driver `595.112.07`; Linux `6.8.0-1069-aws`; Python `3.12.16`; PyTorch `2.8.0+cu128`; CUDA `12.8`; cuDNN `9.23.02` |
| Dtype/backend | FP32 weights and arithmetic, inference mode; TensorFloat-32 (TF32) off for matmul and cuDNN; FP16/BF16 autocast and cuDNN benchmarking off |
| Batch/context | One independent request for latency and quality; at most three predictor states of width 192; current/goal images each `1 x 1 x 3 x 224 x 224` |
| Search budget | 300 candidates per request, 30 elites, 30 rounds; five predicted steps of 10 action features, giving 25 physical 2D commands |
| Checkpoint/data | Released, unchanged LeWorldModel PushT weights; pinned sources and hashes in [upstream.py](upstream.py) and [assets/config.json](assets/config.json); PushT simulator from `stable-worldmodel` `0.0.6` |
| Main latency run | Development scene 12, solver seed 47; one cold call and three warm-up requests per configuration, then 20 measured requests each; shuffled interleaved order seeded with `2027 + 12` |
| Task-success run | 96 starts x solver seeds 42/43/44 = 288 paired cases per policy; batch 1 and context at most 3; one trial per episode/seed/policy; no separate warm-up or timed repetitions |
| Separate batching run | Request batches 1/2/4/8, same FP32/context/search settings per request; request i uses scene `(12+i) mod 24` and seed `47+i`; one cold call, three warm-ups and 15 measured batches per point; its single-request prelude uses 10 repetitions |

Timing includes image transfer, encoding, the full search and returning a CPU
plan. Model loading, image preprocessing and compilation/graph setup are excluded
from warm timing. Each quality case executes 25 commands, replans once, then
executes 25 more. The simulator resets with seed 42 to a recorded start and a goal
from 25 commands later; success means reaching its success region at any command.

Full GPU/runtime metadata above is recorded in the latency and batching
artifacts. The quality run used the same L4/FP32 protocol, but its own JSON
records only the CUDA device and PyTorch version, not a full hardware/revision
snapshot. The companion batching artifact supplies the environment record for
that result directory.

### Profiling / baseline analysis

The baseline makes 150 predictor calls per request, each evaluating 300 candidates.
It repeats recent-history calculations and produces outputs the planner discards.
The compiled baseline still launches one graph per round. These observations
motivate computation reuse and whole-search capture; they do not measure the
current FP32 workload's resource limit.

The matched strict-FP32 comparison in the experiment record measures the original
baseline, compiled baseline, computation-reuse bundle and final CUDA planners.
It establishes their full-request latency differences. A phase-by-phase FP32
profile is still needed to separate prediction, encoding and launch overhead.

### First optimization / implementation status

The same-search bundle (`optimized`) combines the public timeline's 24 October
through 2 November milestones: reuse, output pruning, pointwise fusion and short
attention. The final planners add the 10 November full-round compilation and
13 November whole-search graph milestones. These are implemented; each individual
change's gain has not been isolated. The 16 November Triton experiment showed no
clear additional warm-runtime gain in the saved comparison and was removed in
the 22 November cleanup.

## Setup

Use **Linux, Python 3.12, and an NVIDIA GPU with a CUDA 12.8-compatible driver**.
We tested on an L4 with 24 GB of memory. [requirements.txt](requirements.txt) lists
the dependencies.

```bash
git clone https://github.com/Samarth2709/hpml-lewm-cuda.git
cd hpml-lewm-cuda
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q test_project.py
```

The first test run automatically downloads about 73 MB of weights and upstream
code into `.cache/`, with hash checks. Evaluation inputs are included in `assets/`.
CUDA tests skip without a GPU; CPU tests alone do not validate the GPU code.

## Run the code

From the repository root, with the environment activated, inspect saved results:

```bash
python experiments.py summarize
```

To compare planning time and then check task success in the PushT simulator:

```bash
python experiments.py benchmark --device cuda --scenes 12 --seed 47 \
    --warmups 3 --repeats 20 \
    --configs baseline baseline_compiled optimized_cuda optimized_tree_cuda \
    --output runs/reproduction/benchmark.json

python experiments.py evaluate --device cuda --set eval96 \
    --policies baseline optimized_cuda optimized_tree_cuda \
    --output runs/reproduction/eval96.json
```

Outputs are saved under `runs/reproduction/`. Initial compilation can take over a
minute. Evaluation runs 96 episodes × 3 seeds per planner. Use
`python experiments.py benchmark --help` for batching and other options.

## Experiment record

**Preliminary latency, recorded 22 November:** All rows use the L4, strict FP32/CUDA, batch/context,
three-warm-up and 20-repetition setup above. Times are median full-request latency;
intervals are 95% percentile-bootstrap intervals with 10,000 resamples, and
speedups are ratios of medians using the compiled baseline.

| Planner | Median request time [95% interval] | Speedup vs compiled baseline |
| --- | ---: | ---: |
| Original baseline (`baseline`) | 937.7 ms [932.7, 943.0] | 0.97x |
| Compiled baseline (`baseline_compiled`) | 914.0 ms [910.0, 920.2] | 1.00x |
| Same-search bundle before whole-search capture (`optimized`) | 824.3 ms [823.3, 825.8] | 1.11x |
| Optimized, original search (`optimized_cuda`) | 357.1 ms [351.5, 360.6] | 2.56x |
| Optimized, tree search (`optimized_tree_cuda`) | 152.2 ms [151.1, 154.6] | 6.01x |

`optimized` is the intermediate implementation described above, measured in the
same 22 November run as the other rows. It is not selectable in the final code.

**Preliminary task success:** Using the L4/FP32/CUDA setup above, batch 1, context
at most 3, no separate warm-up, and 96 starts x 3 seeds per policy, the baseline and
`optimized_cuda` both succeeded in **198/288**, matching every outcome. Tree search
succeeded in **196/288**. Its difference from baseline is -0.69 percentage points,
with a 95% episode-cluster bootstrap interval of [-5.90, +4.17] points (96 episode
clusters, 10,000 resamples). This passes the prespecified -7.5-point tolerance on
this cohort; it does not establish identical quality or performance on new tasks.
The expert-action control succeeded in 96/96 starts and the zero-action control
in 0/96; each control runs once per start, without planner warm-up or solver seeds.

The separate 22 November batching artifact measures the 19 November batching
milestone using the settings above and contains
batch latency, requests/second and amortized milliseconds/request. Amortized cost
is not individual request latency. No matched batched baseline was measured.

Raw results: [planning time](results/20261122T182400Z-g6_xlarge-cuda/benchmark.json),
[task success](results/20261122T193543Z-g6_xlarge-cuda/eval96.json), and
[batching](results/20261122T193543Z-g6_xlarge-cuda/batch_sweep.json).

## Code guide

| File | Purpose |
| --- | --- |
| [model.py](model.py) | Model loading and the original planner |
| [optimizations.py](optimizations.py) | Compiled baseline and optimized planners |
| [experiments.py](experiments.py) | Timing, simulator evaluation, and result summaries |
| [upstream.py](upstream.py) | Downloads of the original model and simulator code |
| [test_project.py](test_project.py) | Correctness checks |

## Team

Sayem Kamal, Samarth Kumbla, Samay Lakhani and Jivin Yalamanchili.
Sayem contributed model integration and computation reuse. Samarth contributed
NVIDIA benchmarking, compilation, CUDA graphs, the Triton attempt, and batching.
AI coding assistance was used.
