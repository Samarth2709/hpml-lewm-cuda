"""Pretrained LeWorldModel (PushT), the native CEM planner and the baseline predictor rollout.

Everything here is the unoptimized reference: released weights, strict FP32 and the
native search budget.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import upstream

ASSETS = Path(__file__).resolve().parent / 'assets'
ASSET_SHA256 = {
    'development_scenes.npz': '36f82e755bd1e6bb018a0ca3b244f60ca3ac32320ed334af5216ff5cd81e6235',
    'pusht_96_episode_evaluation_cohort.npz': '0a7e80f12cdbd06c87367b7f3e184271d903e916ff3b95be47a8186486dc488a',
}

# Native CEM budget: 300 candidates, 30 elites, 30 rounds, 5 predicted steps of 10-D action blocks.
CANDIDATES = 300
ELITES = 30
ROUNDS = 30
HORIZON = 5
ACTION_DIM = 10


def asset(name):
    path = ASSETS / name
    upstream.check_sha256(path, ASSET_SHA256[name])
    return path


def config_section(name):
    """Content of one original configuration file stored in assets/config.json."""
    config = json.loads((ASSETS / 'config.json').read_text())
    return config['files'][name]['content']


def public_keys(section):
    """Drop the bookkeeping keys (such as '_target_') from a config section."""
    result = {}
    for key, value in section.items():
        if not key.startswith('_'):
            result[key] = value
    return result


def use_strict_fp32():
    """FP32 everywhere: no TF32 matrix math and no cuDNN autotuning."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_model(device):
    use_strict_fp32()
    jepa, module = upstream.lewm_modules()
    model_config = config_section('model')
    vision_config = config_section('vision_encoder')['config']

    encoder = upstream.Encoder(vision_config)
    predictor = module.ARPredictor(**public_keys(model_config['predictor']))
    action_encoder = module.Embedder(**public_keys(model_config['action_encoder']))
    projector = module.MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    pred_proj = module.MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    model = jepa.JEPA(encoder, predictor, action_encoder, projector, pred_proj)

    state = torch.load(upstream.checkpoint_path(), map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise ValueError('Expected FP32 checkpoint parameters')
    return model.to(device).eval()


def image_pair(scene):
    """Current and goal images of one recorded development scene (0-23), preprocessed on CPU."""
    with np.load(asset('development_scenes.npz')) as data:
        images = data['pixels'][3 * scene:3 * scene + 3]
    pixels = upstream.preprocess(images)
    current = pixels[0:1, None]
    goal = pixels[2:3, None]
    return current, goal


def baseline_rollout(model, z, actions):
    """Released LeWM rollout: re-embed all actions and run the full 3-position window every step."""
    batch, samples, total_steps, _ = actions.shape
    history = z.shape[1]
    width = z.shape[2]
    x = z[:, None].expand(batch, samples, history, width)
    x = x.reshape(batch * samples, history, width).contiguous()
    flat_actions = actions.reshape(batch * samples, total_steps, -1)
    for end in range(history, total_steps + 1):
        action_embedding = model.action_encoder(flat_actions[:, :end])[:, -3:]
        prediction = model.predict(x[:, -3:], action_embedding)[:, -1:]
        if end < total_steps:
            x = torch.cat((x, prediction), dim=1)
    return prediction.reshape(batch, samples, width)


def solve(cost, seed, device):
    """Native CEM: sample around the mean, keep the 30 lowest-cost candidates, refit, repeat."""
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros(1, HORIZON, ACTION_DIM, device=device)
    std = torch.ones_like(mean)
    batch_index = torch.arange(1, device=device).unsqueeze(1).expand(-1, ELITES)
    for _ in range(ROUNDS):
        noise = torch.randn(1, CANDIDATES, HORIZON, ACTION_DIM, generator=generator, device=device)
        candidates = noise * std.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean
        values, indices = torch.topk(cost(candidates), k=ELITES, dim=1, largest=False)
        elites = candidates[batch_index, indices]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1)
    return {
        'actions': mean.cpu().clone(),
        'std': std.cpu().clone(),
        'cost': values.mean(dim=1).cpu().clone(),
    }


@torch.inference_mode()
def request(model, pixels, goal, seed, rollout=baseline_rollout):
    """One planning request: CPU images in, CPU plan out (encode both images, then 30 CEM rounds)."""
    device = next(model.parameters()).device
    z = model.encode({'pixels': pixels.to(device)})['emb']
    target = model.encode({'pixels': goal.to(device)})['emb'][:, -1]

    def cost(candidates):
        prediction = rollout(model, z, candidates)
        return (prediction - target[:, None]).square().sum(-1)

    return solve(cost, seed, device)
