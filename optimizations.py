"""Optimized strict-FP32 planners for LeWorldModel PushT.

Weights, FP32 arithmetic and the 300/30/30/5 search budget stay those of model.py.
Rollout optimizations, in the order they were added:
  1. Growing-prefix K/V cache: steps 1-3 only append one token to the window, so the
     keys/values of earlier tokens are reused. Steps 4-5 shift the 3-position window
     (positions reset), so they are recomputed.
  2. Last-output pruning: the rollout only consumes the newest prediction, so the last
     block of a shifted window computes only the newest query and MLP row.
  3. Pointwise fusion: AdaLN modulation and gated residuals are torch.compile'd kernels.
     Matrix multiplies stay in native PyTorch.
  4. Explicit short attention: with only 1-3 keys, attention is written as compiled FP32
     products, a softmax and sums instead of a scaled_dot_product_attention call.
  5. Shared-prefix tree ('late' layout): CEM candidates are rewritten so the 300 leaves share
     30/30/60/60/300 unique prefixes. Each unique transition is predicted once and its latent
     history and K/V are gathered to its children. This changes the search proposals, not the model.
"""
import time
from functools import lru_cache

import numpy as np
import torch

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


# Unique tree nodes at each of the 5 predicted steps. Every layout ends with 300 leaves.
LAYOUTS = {
    'independent': (300, 300, 300, 300, 300),
    'late': (30, 30, 60, 60, 300),
}


def index_tensor(array, device):
    if array is None:
        return None
    return torch.tensor(array, dtype=torch.int64, device=device)


class TreePlan:
    """Static shared-prefix layout: the parent of every tree node and the node behind every leaf."""

    def __init__(self, layout, device):
        counts = LAYOUTS[layout]
        self.layout = layout
        self.counts = counts

        # parents[d][i] = node at depth d-1 that node i at depth d continues (None: no branching).
        parents = [None]
        for depth in range(1, 5):
            old = counts[depth - 1]
            new = counts[depth]
            if new % old != 0:
                raise ValueError('Every node of a depth must have the same number of children')
            if new == old:
                parents.append(None)
            else:
                parents.append(np.repeat(np.arange(old), new // old))

        # leaf_nodes[d][leaf] = node at depth d on the path to that leaf.
        leaf_nodes = [None, None, None, None, np.arange(300)]
        for depth in range(4, 0, -1):
            if parents[depth] is None:
                leaf_nodes[depth - 1] = leaf_nodes[depth]
            else:
                leaf_nodes[depth - 1] = parents[depth][leaf_nodes[depth]]

        # representatives[d][node] = first leaf below that node.
        # materialize_index[d][leaf] = representative leaf of the leaf's node at depth d.
        representatives = []
        materialize_index = []
        for depth in range(5):
            first_leaf = np.unique(leaf_nodes[depth], return_index=True)[1]
            representatives.append(first_leaf)
            materialize_index.append(first_leaf[leaf_nodes[depth]])

        self.parents = [index_tensor(parent, device) for parent in parents]
        self.leaf_nodes = [index_tensor(nodes, device) for nodes in leaf_nodes]
        self.representatives = [index_tensor(first, device) for first in representatives]
        self.materialize_index = [index_tensor(index, device) for index in materialize_index]

    def materialize(self, samples):
        """Rewrite [R,300,5,10] samples so leaves that share a node at depth d share actions up to d."""
        columns = []
        for depth in range(5):
            column = samples[:, :, depth].index_select(1, self.materialize_index[depth])
            columns.append(column)
        return torch.stack(columns, dim=2)


@lru_cache(maxsize=16)
def tree_plan(layout, device):
    return TreePlan(layout, device)


def gather_cache(cache, index):
    gathered = []
    for k, v in cache:
        gathered.append((k.index_select(0, index), v.index_select(0, index)))
    return gathered


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


def explicit_attention(q, k, v, mask):
    """Attention for 1-3 keys: [B,H,Q,64] queries, [B,H,K,64] keys/values, [Q,K] boolean mask."""
    scale = q.shape[-1] ** -0.5
    scores = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1) * scale
    weights = scores.masked_fill(~mask, float('-inf')).softmax(-1)
    return (weights.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)


# Separate code objects for the dynamic-shape kernels: torch.compile caches (and its recompile
# limit) are per code object, so sharing them with the static kernels would exhaust the limit.
def dynamic_modulation(x, shift, scale):
    return modulation(x, shift, scale)


def dynamic_residual(x, gate, branch):
    return residual(x, gate, branch)


def dynamic_attention(q, k, v, mask):
    return explicit_attention(q, k, v, mask)


@lru_cache(maxsize=2)
def kernels(dynamic):
    """Compiled (modulation, residual, attention). Static for 300-row stages, dynamic for narrower tree stages."""
    if dynamic:
        functions = (dynamic_modulation, dynamic_residual, dynamic_attention)
    else:
        functions = (modulation, residual, explicit_attention)
    compiled = []
    for function in functions:
        compiled.append(torch.compile(function, fullgraph=True, dynamic=dynamic))
    return tuple(compiled)


@lru_cache(maxsize=12)
def attention_mask(queries, keys, device, causal):
    mask = torch.ones(queries, keys, device=device, dtype=torch.bool)
    if causal:
        mask = mask.tril()
    return mask


def short_attention(q, k, v, causal, attend):
    mask = attention_mask(q.shape[2], k.shape[2], q.device, causal)
    return attend(q, k, v, mask)


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


def prefix_step(model, x, c, position, cache, ops):
    """Predict from one new token at prefix position 0/1/2 and append its K/V to the per-layer cache."""
    mod, res, attend = ops
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
        hidden = mod(block.norm1(x), shift, scale)
        q, k, v = attn.to_qkv(attn.norm(hidden)).chunk(3, -1)
        q = split_heads(attn, q)
        k = split_heads(attn, k)
        v = split_heads(attn, v)
        if position > 0:
            old_k, old_v = cache[index]
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        # The newest token may attend to every earlier token (non-causal, one query).
        value = short_attention(q, k, v, False, attend)
        x = res(x, gate, attn.to_out(merge_heads(value, 1)))
        hidden = mod(block.norm2(x), shift_ff, scale_ff)
        x = res(x, gate_ff, block.mlp(hidden))
        cache[index] = (k, v)
    prediction = predicted_latents(model, x)
    if position == 2:
        cache = None  # a window shift follows, so this cache is never read again
    return prediction, cache


def window_attention(attn, x, attend):
    """Causal self-attention over a full 1-3 position window, including both projections."""
    q, k, v = attn.to_qkv(attn.norm(x)).chunk(3, -1)
    q = split_heads(attn, q)
    k = split_heads(attn, k)
    v = split_heads(attn, v)
    value = short_attention(q, k, v, True, attend)
    return attn.to_out(merge_heads(value, x.shape[1]))


def window_step(model, x, c, ops):
    """Predict from a shifted 3-position window. The last block computes only the newest row."""
    mod, res, attend = ops
    predictor = model.predictor
    transformer = predictor.transformer
    position_embedding = predictor.pos_embedding[:, :x.shape[1]]
    x = transformer.input_proj(predictor.dropout(x + position_embedding))
    c = transformer.cond_proj(c)
    for block in transformer.layers[:-1]:
        shift, scale, gate, shift_ff, scale_ff, gate_ff = block.adaLN_modulation(c).chunk(6, -1)
        hidden = mod(block.norm1(x), shift, scale)
        x = res(x, gate, window_attention(block.attn, hidden, attend))
        hidden = mod(block.norm2(x), shift_ff, scale_ff)
        x = res(x, gate_ff, block.mlp(hidden))

    block = transformer.layers[-1]
    shift, scale, gate, shift_ff, scale_ff, gate_ff = block.adaLN_modulation(c).chunk(6, -1)
    attn = block.attn
    hidden = mod(block.norm1(x), shift, scale)
    q, k, v = attn.to_qkv(attn.norm(hidden)).chunk(3, -1)
    # Only the newest row is needed. Its query sees the whole window (non-causal, one query).
    q = split_heads(attn, q[:, -1:])
    k = split_heads(attn, k)
    v = split_heads(attn, v)
    value = short_attention(q, k, v, False, attend)
    x = res(x[:, -1:], gate[:, -1:], attn.to_out(merge_heads(value, 1)))
    hidden = mod(block.norm2(x), shift_ff[:, -1:], scale_ff[:, -1:])
    x = res(x, gate_ff[:, -1:], block.mlp(hidden))
    return predicted_latents(model, x)


def rollout(model, z, actions, layout='independent'):
    """Optimized rollout of 300 candidates from one start. `actions` must follow the layout's ancestry."""
    check_rollout_inputs(z, actions)
    plan = tree_plan(layout, str(z.device))
    flat_actions = actions.reshape(300, 5, 10)
    x = z.expand(plan.counts[0], 1, 192).contiguous()
    cache = None
    for depth in range(5):
        count = plan.counts[depth]
        ops = kernels(count != 300)
        parent = plan.parents[depth]
        if parent is not None:
            # Branching: children inherit the latent history and cached keys/values of their parent.
            x = x.index_select(0, parent)
            if cache is not None:
                cache = gather_cache(cache, parent)
        prefix = flat_actions[:, :depth + 1]
        if count != 300:
            # All leaves below a node share its action prefix, so one representative is embedded.
            prefix = prefix.index_select(0, plan.representatives[depth])
        c = model.action_encoder(prefix)[:, -3:]
        if depth < 3:
            prediction, cache = prefix_step(model, x[:, -1:], c[:, -1:], depth, cache, ops)
        else:
            # Positions shift from here on, so the cached keys/values would be invalid.
            prediction = window_step(model, x[:, -3:], c, ops)
        if depth < 4:
            x = torch.cat((x, prediction), dim=1)
    return prediction.reshape(1, 300, 192)
