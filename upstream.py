"""Pinned upstream code: LeWorldModel classes, the ViT encoder and the PushT simulator.

No upstream source file is stored in this repository. Each file is downloaded once
into .cache/, checked against a pinned SHA-256 and then loaded unchanged.
Licenses and notices: THIRD_PARTY_NOTICES.txt.
"""
import ast
import collections
import hashlib
import importlib.util
import math
import os
import sys
import types
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache'

# LeWorldModel source (MIT, Lucas Maes): https://github.com/lucas-maes/le-wm
LEWM_URL = 'https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/'
LEWM_FILES = {
    'jepa.py': '41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96',
    'module.py': '0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620',
}

# LeWM PushT checkpoint (MIT): https://huggingface.co/quentinll/lewm-pusht
WEIGHTS_URL = ('https://huggingface.co/quentinll/lewm-pusht/resolve/'
               '22b330c28c27ead4bfd1888615af1340e3fe9052/weights.pt')
WEIGHTS_SHA256 = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'

# Hugging Face Transformers v5.3.0 ViT model file (Apache-2.0).
VIT_URL = ('https://raw.githubusercontent.com/huggingface/transformers/'
           'aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/vit/modeling_vit.py')
VIT_SHA256 = '34c5d1689afd7e0002990a300565e0278ac87a64634e631b60ceb9f30adfe99e'
VIT_CLASSES = [
    'ViTEmbeddings', 'ViTPatchEmbeddings', 'ViTSelfAttention', 'ViTSelfOutput', 'ViTAttention',
    'ViTIntermediate', 'ViTOutput', 'ViTLayer', 'ViTEncoder',
]

# stable-worldmodel 0.0.6 wheel (MIT). Only the PushT environment files are used.
WHEEL_URL = ('https://files.pythonhosted.org/packages/77/ac/9618ee77bf0ed0180c67a0645dfb42215f54197de347235c831f65ea6363/'
             'stable_worldmodel-0.0.6-py3-none-any.whl')
WHEEL_SHA256 = '90601307b3430436d617a5134579089414a1f98eab1dfb6aeabf36ea7f46ff4d'
SIMULATOR_FILES = {
    'stable_worldmodel/__init__.py': '098841b3d30e15ebc9ef7d6f4448b971c33369b93d9cd3a4a7b7b94e548f223a',
    'stable_worldmodel/spaces.py': '1c4346e3498c70ef0f4ea3f8320687e52fd601d973907703bdcc5f90bf09b220',
    'stable_worldmodel/utils.py': '58b5df5e11222ca1ecf07d913939c219b34721644a1d285a9143c440ff2f9ad5',
    'stable_worldmodel/envs/__init__.py': '3ae8cbbeecfb0dc52886dd6a1defec6caa3bf5e8df5d7d8f15ee559f4e584d95',
    'stable_worldmodel/envs/utils.py': '38314dcabc556c954d64b1c2a1b558e32b48a0db092657dd745dcb960c9cf6f0',
    'stable_worldmodel/envs/pusht/env.py': 'd8d0de35aaab5b846db4e79b0fbfd6b17375178cce40a25df5301c8030ca6d68',
}


def check_sha256(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path}: got {actual}, expected {expected}')


def fetch(url, name, sha256):
    """Download `url` to .cache/<name> once and check its SHA-256 on every use."""
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {url}', flush=True)
        partial = path.with_name(path.name + '.part')
        urllib.request.urlretrieve(url, partial)
        check_sha256(partial, sha256)
        partial.rename(path)
    check_sha256(path, sha256)
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lewm_modules():
    """Return the pinned LeWorldModel `jepa` and `module` source modules."""
    if 'lewm_jepa' in sys.modules:
        return sys.modules['lewm_jepa'], sys.modules['lewm_module']
    jepa_path = fetch(LEWM_URL + 'jepa.py', 'le-wm/jepa.py', LEWM_FILES['jepa.py'])
    module_path = fetch(LEWM_URL + 'module.py', 'le-wm/module.py', LEWM_FILES['module.py'])
    jepa = load_module('lewm_jepa', jepa_path)
    module = load_module('lewm_module', module_path)
    return jepa, module


def checkpoint_path():
    return fetch(WEIGHTS_URL, 'lewm-pusht/weights.pt', WEIGHTS_SHA256)


# ---------------------------------------------------------------------------
# ViT encoder: nine classes taken unchanged from the pinned Transformers file.
# Framework helpers they refer to are replaced by the small stand-ins below.
# ---------------------------------------------------------------------------

def sdpa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Inference-only stand-in for the Transformers 'sdpa' attention function."""
    if module.training or dropout != 0:
        raise ValueError('Encoder attention shim is inference-only')
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                         dropout_p=0.0, is_causal=False, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def attention_interface(name, default):
    if name == 'sdpa':
        return sdpa_attention
    return default


def vit_classes():
    """Execute only the selected ViT class definitions from the pinned source file."""
    path = fetch(VIT_URL, 'transformers-5.3.0/modeling_vit.py', VIT_SHA256)
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in VIT_CLASSES:
            selected.append(node)
    found = {node.name for node in selected}
    if found != set(VIT_CLASSES):
        raise ValueError('Missing reviewed encoder class')

    future_import = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future_import] + selected, type_ignores=[])
    module = ast.fix_missing_locations(module)
    namespace = {
        'torch': torch,
        'nn': nn,
        'collections': collections,
        'math': math,
        'torch_int': int,
        'ViTConfig': SimpleNamespace,
        'GradientCheckpointingLayer': nn.Module,
        'ACT2FN': {'gelu': nn.GELU()},
        'BaseModelOutput': SimpleNamespace,
        'ALL_ATTENTION_FUNCTIONS': SimpleNamespace(get_interface=attention_interface),
        'eager_attention_forward': None,
    }
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class Encoder(nn.Module):
    """ViT-tiny image encoder (224x224 input, patch 14) with the checkpoint's module names."""

    def __init__(self, vision_config):
        super().__init__()
        classes = vit_classes()
        config = SimpleNamespace(**vision_config, _attn_implementation='sdpa')
        shape = (config.image_size, config.patch_size, config.hidden_size, config.num_hidden_layers)
        if shape != (224, 14, 192, 12):
            raise ValueError('Adapter only supports the audited fixed-224 encoder')
        self.embeddings = classes['ViTEmbeddings'](config, use_mask_token=False)
        self.encoder = classes['ViTEncoder'](config)
        self.layernorm = nn.LayerNorm(192, eps=config.layer_norm_eps)

    def forward(self, pixels, **kwargs):
        if pixels.ndim != 4 or pixels.shape[1:] != (3, 224, 224):
            raise ValueError('Expected normalized [B,3,224,224] images')
        x = self.embeddings(pixels, interpolate_pos_encoding=True)
        hidden = self.encoder(x).last_hidden_state
        return SimpleNamespace(last_hidden_state=self.layernorm(hidden))


def preprocess(image_arrays):
    """uint8 HWC RGB images -> normalized FP32 [N,3,224,224] tensor (LeWM image transform)."""
    for image in image_arrays:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError('Expected uint8 HWC RGB images, not float 0..255 or CHW')
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Expected uint8 HWC RGB images, not float 0..255 or CHW')
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Resize(224, antialias=True),
    ])
    tensors = [transform(image) for image in image_arrays]
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# PushT simulator from the pinned stable-worldmodel wheel.
# A declared default-variation reset with an explicit state, not original seeded replay.
# ---------------------------------------------------------------------------

PUSHT = None


def simulator_source():
    """Extract the pinned simulator files from the wheel into .cache/ and check each one."""
    wheel = fetch(WHEEL_URL, 'stable_worldmodel-0.0.6-py3-none-any.whl', WHEEL_SHA256)
    folder = CACHE / 'stable_worldmodel-0.0.6'
    with zipfile.ZipFile(wheel) as archive:
        for name, sha256 in SIMULATOR_FILES.items():
            target = folder / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            check_sha256(target, sha256)
    return folder / 'stable_worldmodel'


def namespace_package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    return package


def load_pusht():
    """Load the original PushT class without importing the rest of stable-worldmodel."""
    global PUSHT
    if PUSHT is not None:
        return PUSHT
    if 'stable_worldmodel' in sys.modules:
        raise RuntimeError('A different stable_worldmodel package is already loaded; use a fresh process')
    source = simulator_source()
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pymunk
    if pymunk.version != '7.2.0':
        raise RuntimeError(f'Expected Pymunk 7.2.0, got {pymunk.version}')

    package = namespace_package('stable_worldmodel', source)
    package.utils = load_module('stable_worldmodel.utils', source / 'utils.py')
    package.spaces = load_module('stable_worldmodel.spaces', source / 'spaces.py')
    envs = namespace_package('stable_worldmodel.envs', source / 'envs')
    package.envs = envs
    envs.utils = load_module('stable_worldmodel.envs.utils', source / 'envs' / 'utils.py')
    envs.pusht = namespace_package('stable_worldmodel.envs.pusht', source / 'envs' / 'pusht')
    env_module = load_module('stable_worldmodel.envs.pusht.env', source / 'envs' / 'pusht' / 'env.py')
    PUSHT = env_module.PushT
    return PUSHT


def check_state(value, name):
    state = np.asarray(value, dtype=np.float64).copy()
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f'{name} must be a finite seven-value state')
    if not 0 <= state[4] < 2 * np.pi:
        raise ValueError(f'{name} angle must be in [0, 2*pi), as in source observations')
    return state


class PushTAdapter:
    """Original reset/step/render, plus input checks and RGB pixels in each observation."""

    def __init__(self):
        PushT = load_pusht()
        self.env = PushT(resolution=224, render_mode='rgb_array', relative=True,
                         with_target=True, render_action=False)
        self.goal_state = None

    def reset(self, state, goal, seed=42):
        state = check_state(state, 'state')
        goal = check_state(goal, 'goal')
        self.goal_state = goal.copy()
        options = {'state': state, 'goal_state': goal, 'variation': []}
        observation, info = self.env.reset(seed=int(seed), options=options)
        observation = dict(observation)
        observation['pixels'] = self.render()
        return observation, info

    def step(self, action):
        if self.goal_state is None:
            raise RuntimeError('reset must be called before step')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('action must contain two finite raw source-action coordinates')
        observation, reward, terminated, truncated, info = self.env.step(action.copy())
        observation = dict(observation)
        observation['pixels'] = self.render()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.goal_state is None:
            raise RuntimeError('reset must be called before render')
        image = self.env.render()
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise RuntimeError('Unexpected source render shape or dtype')
        return image.copy()

    def close(self):
        self.env.close()


def create_simulator():
    return PushTAdapter()
