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
  6. Shared sliding window: at a branching window step (step 5 in 'late'), siblings share window
     slots 0-1, so those are computed once per parent and each leaf runs only its newest token.
  7. Compiled CEM round (CUDA): sampling, the rollout, the goal cost, top-k and the mean/std
     update of one round form a single torch.compile function. The rollout then uses plain
     PyTorch operations (raw_ops=True) so Inductor can fuse across the whole round.
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

        # first_child[d][p] = first node at depth d whose parent is node p at depth d-1.
        first_child = []
        for parent in parents:
            if parent is None:
                first_child.append(None)
            else:
                first_child.append(np.unique(parent, return_index=True)[1])

        self.parents = [index_tensor(parent, device) for parent in parents]
        self.leaf_nodes = [index_tensor(nodes, device) for nodes in leaf_nodes]
        self.representatives = [index_tensor(first, device) for first in representatives]
        self.materialize_index = [index_tensor(index, device) for index in materialize_index]
        self.first_child = [index_tensor(first, device) for first in first_child]

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


# Plain versions, traced by an outer torch.compile of the whole CEM round.
RAW_OPS = (modulation, residual, explicit_attention)


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


def shared_window_step(model, x, c, parent, first_child, parent_ops, ops):
    """Window step where siblings (same parent) share window slots 0-1: compute those once per parent.

    Attention is causal and action embeddings are per time step, so slots 0-1 never see slot 2
    and are identical across siblings. Each child then runs only its newest token, attending to
    its parent's slot 0-1 keys/values plus its own. x: [P,3,D] parent windows, c: [N,3,D] child
    action embeddings.
    """
    parent_mod, parent_res, parent_attend = parent_ops
    mod, res, attend = ops
    predictor = model.predictor
    transformer = predictor.transformer
    position = predictor.pos_embedding[:, :3]
    parent_hidden = transformer.input_proj(predictor.dropout(x[:, :2] + position[:, :2]))
    child_hidden = transformer.input_proj(predictor.dropout(x[:, 2:] + position[:, 2:]))
    child_hidden = child_hidden.index_select(0, parent)
    first_child_c = c.index_select(0, first_child)
    parent_c = transformer.cond_proj(first_child_c[:, :2])
    child_c = transformer.cond_proj(c[:, 2:])
    layers = transformer.layers
    for index, block in enumerate(layers):
        attn = block.attn
        last = index == len(layers) - 1
        parent_terms = block.adaLN_modulation(parent_c).chunk(6, -1)
        p_shift, p_scale, p_gate, p_shift_ff, p_scale_ff, p_gate_ff = parent_terms
        shift, scale, gate, shift_ff, scale_ff, gate_ff = block.adaLN_modulation(child_c).chunk(6, -1)

        parent_in = attn.norm(parent_mod(block.norm1(parent_hidden), p_shift, p_scale))
        parent_q, parent_k, parent_v = attn.to_qkv(parent_in).chunk(3, -1)
        parent_q = split_heads(attn, parent_q)
        parent_k = split_heads(attn, parent_k)
        parent_v = split_heads(attn, parent_v)
        child_in = attn.norm(mod(block.norm1(child_hidden), shift, scale))
        q, k, v = attn.to_qkv(child_in).chunk(3, -1)
        q = split_heads(attn, q)
        k = split_heads(attn, k)
        v = split_heads(attn, v)
        k = torch.cat((parent_k.index_select(0, parent), k), dim=2)
        v = torch.cat((parent_v.index_select(0, parent), v), dim=2)

        if not last:  # the last block needs the parent slots only as keys/values
            value = short_attention(parent_q, parent_k, parent_v, True, parent_attend)
            parent_hidden = parent_res(parent_hidden, p_gate, attn.to_out(merge_heads(value, 2)))
            hidden = parent_mod(block.norm2(parent_hidden), p_shift_ff, p_scale_ff)
            parent_hidden = parent_res(parent_hidden, p_gate_ff, block.mlp(hidden))

        value = short_attention(q, k, v, False, attend)
        child_hidden = res(child_hidden, gate, attn.to_out(merge_heads(value, 1)))
        hidden = mod(block.norm2(child_hidden), shift_ff, scale_ff)
        child_hidden = res(child_hidden, gate_ff, block.mlp(hidden))
    return predicted_latents(model, child_hidden)


def rollout(model, z, actions, layout='independent', raw_ops=False, plan=None, use_shared_window=True):
    """Optimized rollout of 300 candidates from one start. `actions` must follow the layout's ancestry."""
    check_rollout_inputs(z, actions)
    if plan is None:
        plan = tree_plan(layout, str(z.device))
    flat_actions = actions.reshape(300, 5, 10)
    x = z.expand(plan.counts[0], 1, 192).contiguous()
    cache = None
    for depth in range(5):
        count = plan.counts[depth]
        if raw_ops:
            ops = RAW_OPS
            parent_ops = RAW_OPS
        else:
            ops = kernels(count != 300)
            parent_ops = kernels(True)
        parent = plan.parents[depth]
        shared_window = use_shared_window and depth >= 3 and parent is not None
        if parent is not None and not shared_window:
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
        elif shared_window:
            prediction = shared_window_step(model, x[:, -3:], c, parent, plan.first_child[depth],
                                            parent_ops, ops)
            x = x.index_select(0, parent)
        else:
            # Positions shift from here on, so the cached keys/values would be invalid.
            prediction = window_step(model, x[:, -3:], c, ops)
        if depth < 4:
            x = torch.cat((x, prediction), dim=1)
    return prediction.reshape(1, 300, 192)


def compile_cem_round(model, layout, use_shared_window=True):
    """One complete CEM round as a single torch.compile function with static shapes.

    Inductor fuses the normalization, modulation, GELU, residual and CEM elementwise work and
    autotunes the FP32 matrix multiplies. No CUDA graphs are used inside the compiled code.
    """
    device = next(model.parameters()).device
    plan = tree_plan(layout, str(device))
    batch_index = torch.arange(1, device=device).unsqueeze(1).expand(-1, M.ELITES)

    def cem_round(z, target, mean, std, noise):
        candidates = noise * std.unsqueeze(1) + mean.unsqueeze(1)
        # Same as `candidates[:, 0] = mean`, written without an in-place write for the compiler.
        candidates = torch.cat((mean.unsqueeze(1), candidates[:, 1:]), dim=1)
        if layout != 'independent':
            candidates = plan.materialize(candidates)
        prediction = rollout(model, z, candidates, layout, raw_ops=True, plan=plan,
                             use_shared_window=use_shared_window)
        costs = (prediction - target[:, None]).square().sum(-1)
        values, indices = torch.topk(costs, k=M.ELITES, dim=1, largest=False)
        elites = candidates[batch_index, indices]
        return elites.mean(dim=1), elites.std(dim=1), values

    return torch.compile(cem_round, mode='max-autotune-no-cudagraphs', dynamic=False)


class CompiledRoundPlanner:
    """Native CEM where every round runs as one compiled function (see compile_cem_round).

    The 30 rounds are still launched one by one from Python. The noise comes from the same
    generator calls as model.solve, so the random stream is unchanged.
    """

    def __init__(self, model, layout='independent', use_shared_window=True):
        self.model = model
        self.layout = layout
        self.use_shared_window = use_shared_window
        self.compiled_round = None

    @torch.inference_mode()
    def request(self, pixels, goal, seed):
        device = next(self.model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('CompiledRoundPlanner requires a CUDA model')
        if self.compiled_round is None:
            self.compiled_round = compile_cem_round(self.model, self.layout, self.use_shared_window)
        z = self.model.encode({'pixels': pixels.to(device)})['emb']
        target = self.model.encode({'pixels': goal.to(device)})['emb'][:, -1]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        mean = torch.zeros(1, M.HORIZON, M.ACTION_DIM, device=device)
        std = torch.ones_like(mean)
        for _ in range(M.ROUNDS):
            noise = torch.randn(1, M.CANDIDATES, M.HORIZON, M.ACTION_DIM, generator=generator, device=device)
            mean, std, values = self.compiled_round(z, target, mean, std, noise)
        return {
            'actions': mean.cpu().clone(),
            'std': std.cpu().clone(),
            'cost': values.mean(dim=1).cpu().clone(),
        }
