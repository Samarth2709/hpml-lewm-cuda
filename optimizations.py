"""Optimized strict-FP32 planners for LeWorldModel PushT.

Weights, FP32 arithmetic and the 300/30/30/5 search budget stay those of model.py.
"""
import time

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
