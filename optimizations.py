"""Optimized strict-FP32 planners for LeWorldModel PushT.

Weights, FP32 arithmetic and the 300/30/30/5 search budget stay those of model.py.
Rollout optimizations, in the order they were added:
  1. Growing-prefix K/V cache: steps 1-3 only append one token to the window, so the
     keys/values of earlier tokens are reused. Steps 4-5 shift the 3-position window
     (positions reset), so they are recomputed.
"""
import time

import torch
from torch.nn import functional as F

import model as M


class CompiledGraphBaseline:
    """Strong CUDA reference: the unchanged baseline rollout and goal cost, compiled with
    torch.compile(mode='max-autotune-no-cudagraphs') and replayed as one CUDA graph per CEM round.
    """

    def __init__(self, model):
        self.model = model
        self.graph = None
        self.build_ms = None

    def build(self, z, target, actions):
        start = time.perf_counter()

        def cost_function(z_in, target_in, actions_in):
            prediction = M.baseline_rollout(self.model, z_in, actions_in)
            return (prediction - target_in[:, None]).square().sum(-1)

        compiled = torch.compile(cost_function, mode='max-autotune-no-cudagraphs', dynamic=False)
        # Static input buffers: every round copies its inputs here and replays the graph.
        self.static_z = z.clone()
        self.static_target = target.clone()
        self.static_actions = actions.clone()
        for _ in range(3):
            compiled(self.static_z, self.static_target, self.static_actions)
        torch.cuda.synchronize(z.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_cost = compiled(self.static_z, self.static_target, self.static_actions)
        torch.cuda.synchronize(z.device)
        self.build_ms = (time.perf_counter() - start) * 1e3

    @torch.inference_mode()
    def request(self, pixels, goal, seed):
        device = next(self.model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('CompiledGraphBaseline requires a CUDA model')
        z = self.model.encode({'pixels': pixels.to(device)})['emb']
        target = self.model.encode({'pixels': goal.to(device)})['emb'][:, -1]

        def cost(candidates):
            if self.graph is None:
                self.build(z, target, candidates)
            self.static_z.copy_(z)
            self.static_target.copy_(target)
            self.static_actions.copy_(candidates)
            self.graph.replay()
            return self.static_cost

        return M.solve(cost, seed, device)


def check_rollout_inputs(z, actions):
    if torch.is_grad_enabled():
        raise ValueError('The optimized rollout runs in inference mode only')
    if z.ndim != 3 or z.shape[1:] != (1, 192) or z.shape[0] != 1:
        raise ValueError(f'Expected z of shape [1,1,192], got {tuple(z.shape)}')
    if actions.ndim != 4 or actions.shape != (1, 300, 5, 10):
        raise ValueError(f'Expected actions of shape [1,300,5,10], got {tuple(actions.shape)}')


def modulation(x, shift, scale):
    return x * (1 + scale) + shift


def residual(x, gate, branch):
    return x + gate * branch


def split_heads(attn, t):
    """[B, T, heads*64] -> [B, heads, T, 64]"""
    return t.reshape(t.shape[0], t.shape[1], attn.heads, -1).transpose(1, 2)


def merge_heads(value, length):
    """[B, heads, T, 64] -> [B, T, heads*64]"""
    return value.transpose(1, 2).reshape(value.shape[0], length, -1)


def predicted_latents(model, hidden):
    """Final norm and output projections: [B,1,hidden] states -> [B,1,192] predicted latents."""
    transformer = model.predictor.transformer
    hidden = transformer.output_proj(transformer.norm(hidden))
    rows = hidden.shape[0]
    flat = hidden.reshape(-1, hidden.shape[-1])
    return model.pred_proj(flat).reshape(rows, 1, -1)


def prefix_step(model, x, c, position, cache):
    """Predict from one new token at prefix position 0/1/2 and append its K/V to the per-layer cache."""
    predictor = model.predictor
    transformer = predictor.transformer
    position_embedding = predictor.pos_embedding[:, position:position + 1]
    x = transformer.input_proj(predictor.dropout(x + position_embedding))
    c = transformer.cond_proj(c)
    if position == 0:
        cache = [None] * len(transformer.layers)
    for index, block in enumerate(transformer.layers):
        shift, scale, gate, shift_ff, scale_ff, gate_ff = block.adaLN_modulation(c).chunk(6, -1)
        attn = block.attn
        hidden = modulation(block.norm1(x), shift, scale)
        q, k, v = attn.to_qkv(attn.norm(hidden)).chunk(3, -1)
        q = split_heads(attn, q)
        k = split_heads(attn, k)
        v = split_heads(attn, v)
        if position > 0:
            old_k, old_v = cache[index]
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        # The newest token may attend to every earlier token, so no mask is needed.
        value = F.scaled_dot_product_attention(q, k, v)
        x = residual(x, gate, attn.to_out(merge_heads(value, 1)))
        hidden = modulation(block.norm2(x), shift_ff, scale_ff)
        x = residual(x, gate_ff, block.mlp(hidden))
        cache[index] = (k, v)
    prediction = predicted_latents(model, x)
    if position == 2:
        cache = None  # a window shift follows, so this cache is never read again
    return prediction, cache


def rollout(model, z, actions):
    """Optimized rollout of 300 candidates from one start. Same result as model.baseline_rollout."""
    check_rollout_inputs(z, actions)
    flat_actions = actions.reshape(300, 5, 10)
    x = z.expand(300, 1, 192).contiguous()
    cache = None
    for depth in range(5):
        c = model.action_encoder(flat_actions[:, :depth + 1])[:, -3:]
        if depth < 3:
            prediction, cache = prefix_step(model, x[:, -1:], c[:, -1:], depth, cache)
        else:
            # Positions shift from here on, so the cached keys/values would be invalid.
            prediction = model.predict(x[:, -3:], c)[:, -1:]
        if depth < 4:
            x = torch.cat((x, prediction), dim=1)
    return prediction.reshape(1, 300, 192)
