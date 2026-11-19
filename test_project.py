"""Tests: pinned assets, statistics, model loading, the PushT simulator and CUDA planners.

Run: python -m pytest -q test_project.py
The first run downloads the pinned checkpoint and upstream sources into .cache/.
"""
import hashlib
import importlib.util
import json
import unittest
from functools import lru_cache

import numpy as np
import torch

import experiments as E
import model as M
import optimizations as O
import upstream

HAS_SIMULATOR_DEPS = importlib.util.find_spec('pymunk') is not None


@lru_cache(maxsize=1)
def cpu_model():
    torch.set_num_threads(2)
    return M.load_model('cpu')


class AssetTest(unittest.TestCase):
    def test_npz_hashes_and_shapes(self):
        with np.load(M.asset('development_scenes.npz')) as data:
            self.assertEqual(data['pixels'].shape, (72, 224, 224, 3))
            self.assertEqual(data['states'].shape, (24, 26, 7))
        with np.load(M.asset('pusht_96_episode_evaluation_cohort.npz')) as data:
            self.assertEqual(data['states'].shape, (96, 26, 7))
            self.assertEqual(data['raw_actions'].shape, (96, 25, 2))

    def test_config_reproduces_original_files(self):
        config = json.loads((M.ASSETS / 'config.json').read_text())
        for name, entry in config['files'].items():
            text = json.dumps(entry['content'], indent=entry['json_indent'])
            if entry['trailing_newline']:
                text += '\n'
            digest = hashlib.sha256(text.encode()).hexdigest()
            self.assertEqual(digest, entry['sha256'], name)

    def test_simulator_pins_match_recorded_receipt(self):
        receipt = M.config_section('simulator_wheel')
        self.assertEqual(receipt['source_url'], upstream.WHEEL_URL)
        self.assertEqual(receipt['sha256'], upstream.WHEEL_SHA256)
        self.assertEqual(M.config_section('simulator_files'), upstream.SIMULATOR_FILES)


class StatsTest(unittest.TestCase):
    def test_mcnemar_and_quality_gate(self):
        self.assertEqual(E.mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(E.mcnemar_exact_p(1, 3), 0.625, places=6)
        records = []
        for row in range(20):
            for seed in (42, 43, 44):
                ok = row % 2 == 0
                records.append({'row': row, 'seed': seed, 'policy': 'baseline', 'success': ok})
                records.append({'row': row, 'seed': seed, 'policy': 'good', 'success': ok})
                records.append({'row': row, 'seed': seed, 'policy': 'bad', 'success': False})
        good = E.quality_gate(records, 'baseline', 'good', resamples=1000)
        bad = E.quality_gate(records, 'baseline', 'bad', resamples=1000)
        self.assertTrue(good['passed'])
        self.assertEqual(good['difference'], 0.0)
        self.assertEqual(good['ci_95'], [0.0, 0.0])
        self.assertFalse(bad['passed'])
        self.assertLess(bad['ci_95'][0], -0.075)

    def test_bootstrap_timing_and_speedup(self):
        reference = [100.0, 102.0, 98.0, 101.0, 99.0]
        variant = [50.0, 51.0, 49.0, 50.5, 49.5]
        summary = E.summarize_timings(reference, resamples=1000)
        self.assertAlmostEqual(summary['median_ms'], 100.0)
        self.assertLessEqual(summary['median_ci_ms'][0], summary['median_ms'])
        self.assertGreaterEqual(summary['median_ci_ms'][1], summary['median_ms'])
        speedup = E.bootstrap_speedup_ci(reference, variant, resamples=1000)
        self.assertAlmostEqual(speedup['ratio'], 2.0, places=5)
        self.assertGreater(speedup['ci'][0], 1.9)


class ModelTest(unittest.TestCase):
    @torch.inference_mode()
    def test_checkpoint_encoder_and_baseline_rollout(self):
        model = cpu_model()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        pixels, goal = M.image_pair(12)
        self.assertEqual(tuple(pixels.shape), (1, 1, 3, 224, 224))
        z = model.encode({'pixels': pixels})['emb']
        self.assertEqual(tuple(z.shape), (1, 1, 192))
        self.assertTrue(torch.isfinite(z).all())
        gen = torch.Generator().manual_seed(0)
        actions = torch.randn(1, 300, 5, 10, generator=gen)
        prediction = M.baseline_rollout(model, z, actions)
        self.assertEqual(tuple(prediction.shape), (1, 300, 192))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_cem_is_deterministic_for_a_seed(self):
        target = torch.linspace(-1, 1, M.HORIZON * M.ACTION_DIM).reshape(1, 1, M.HORIZON, M.ACTION_DIM)

        def cost(candidates):
            return (candidates - target).square().sum((-1, -2))

        first = M.solve(cost, 47, 'cpu')
        second = M.solve(cost, 47, 'cpu')
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertEqual(tuple(first['actions'].shape), (1, M.HORIZON, M.ACTION_DIM))
        error_before = target.square().sum()
        error_after = (first['actions'] - target[0]).square().sum()
        self.assertLess(float(error_after), float(error_before))


class TreePlanTest(unittest.TestCase):
    def test_leaves_share_exactly_the_declared_prefixes(self):
        samples = torch.arange(15000, dtype=torch.float32).reshape(1, 300, 5, 10)
        for layout in O.LAYOUTS:
            plan = O.TreePlan(layout, 'cpu')
            actions = plan.materialize(samples)[0]
            for depth in range(5):
                prefixes = actions[:, :depth + 1]
                unique = prefixes.index_select(0, plan.representatives[depth])
                self.assertEqual(len(torch.unique(prefixes.flatten(1), dim=0)), plan.counts[depth])
                self.assertTrue(torch.equal(prefixes, unique.index_select(0, plan.leaf_nodes[depth])))
        independent = O.TreePlan('independent', 'cpu').materialize(samples)
        self.assertTrue(torch.equal(independent, samples))

    def test_batched_roots_repeat_the_single_root_plan(self):
        samples = torch.randn(2, 300, 5, 10, generator=torch.Generator().manual_seed(3))
        single = O.TreePlan('late', 'cpu')
        batched = O.TreePlan('late', 'cpu', batch_size=2)
        self.assertEqual(batched.counts, (60, 60, 120, 120, 600))
        both = batched.materialize(samples)
        for root in range(2):
            self.assertTrue(torch.equal(both[root:root + 1], single.materialize(samples[root:root + 1])))


class RolloutTest(unittest.TestCase):
    @torch.inference_mode()
    def test_optimized_rollout_matches_baseline(self):
        model = cpu_model()
        gen = torch.Generator().manual_seed(0)
        z = torch.randn(1, 1, 192, generator=gen)
        for layout in ('independent', 'late'):
            samples = torch.randn(1, 300, 5, 10, generator=gen)
            actions = O.TreePlan(layout, 'cpu').materialize(samples)
            reference = M.baseline_rollout(model, z, actions)
            optimized = O.rollout(model, z, actions, layout)
            torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-5)
            plain = O.rollout(model, z, actions, layout, raw_ops=True)
            torch.testing.assert_close(plain, reference, rtol=1e-5, atol=1e-5)

    @torch.inference_mode()
    def test_batched_roots_match_baseline(self):
        model = cpu_model()
        gen = torch.Generator().manual_seed(1)
        z = torch.randn(2, 1, 192, generator=gen)
        for layout in ('independent', 'late'):
            samples = torch.randn(2, 300, 5, 10, generator=gen)
            actions = O.TreePlan(layout, 'cpu', batch_size=2).materialize(samples)
            reference = M.baseline_rollout(model, z, actions)
            optimized = O.rollout(model, z, actions, layout, raw_ops=True)
            torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-5)

    @torch.inference_mode()
    def test_shared_window_matches_separate_windows(self):
        model = cpu_model()
        gen = torch.Generator().manual_seed(2)
        z = torch.randn(1, 1, 192, generator=gen)
        actions = O.TreePlan('late', 'cpu').materialize(torch.randn(1, 300, 5, 10, generator=gen))
        shared = O.rollout(model, z, actions, 'late', use_shared_window=True)
        separate = O.rollout(model, z, actions, 'late', use_shared_window=False)
        torch.testing.assert_close(shared, separate, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(HAS_SIMULATOR_DEPS, 'PushT simulator dependencies are not installed')
class SimulatorTest(unittest.TestCase):
    def test_expert_and_zero_controls(self):
        with np.load(M.asset('pusht_96_episode_evaluation_cohort.npz')) as archive:
            data = {'states': archive['states'], 'raw_actions': archive['raw_actions']}
        expert = E.run_control(data, 0, 'expert_25')
        zero = E.run_control(data, 0, 'zero_50')
        self.assertTrue(expert['success'])
        self.assertGreaterEqual(expert['first_success'], 20)
        self.assertFalse(zero['success'])


def assert_plans_close(first, second):
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=1e-4, atol=1e-4)


@lru_cache(maxsize=1)
def cuda_model():
    return M.load_model('cuda')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class CudaTest(unittest.TestCase):
    def test_compiled_baseline_matches_eager_baseline(self):
        model = cuda_model()
        pair = M.image_pair(12)
        reference = M.request(model, *pair, 47)
        compiled = O.CompiledGraphBaseline(model).request(*pair, 47)
        assert_plans_close(compiled, reference)

    def test_whole_search_graph_matches_eager_search(self):
        model = cuda_model()
        pair = M.image_pair(12)
        same_search = M.request(model, *pair, 47)
        planner = O.WholeSearchCUDAGraph(model, 'independent')
        first = planner.request(*pair, 47)
        assert_plans_close(first, same_search)
        second = planner.request(*pair, 47)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
        plan = O.tree_plan('late', 'cuda')
        tree_search = M.request(model, *pair, 47, rollout=O.rollout, plan=plan)
        assert_plans_close(O.WholeSearchCUDAGraph(model, 'late').request(*pair, 47), tree_search)
        no_shared = O.WholeSearchCUDAGraph(model, 'late', use_shared_window=False)
        assert_plans_close(no_shared.request(*pair, 47), tree_search)

    def test_batched_graph_matches_single_requests(self):
        model = cuda_model()
        pairs = [M.image_pair(12), M.image_pair(13)]
        pixels = torch.cat([pairs[0][0], pairs[1][0]], dim=0)
        goals = torch.cat([pairs[0][1], pairs[1][1]], dim=0)
        for layout in ('independent', 'late'):
            batched = O.WholeSearchCUDAGraph(model, layout, batch_size=2).request(pixels, goals, [47, 48])
            single = O.WholeSearchCUDAGraph(model, layout)
            for index in range(2):
                one = single.request(*pairs[index], 47 + index)
                for key in one:
                    torch.testing.assert_close(batched[key][index:index + 1], one[key], rtol=1e-4, atol=1e-4)

    @torch.inference_mode()
    def test_triton_short_attention_matches_explicit(self):
        gen = torch.Generator(device='cuda').manual_seed(7)
        shapes = [(1, 1, False), (1, 2, False), (1, 3, False), (2, 2, True), (3, 3, True),
                  (2, 2, False), (3, 3, False)]
        for batch in (30, 60, 300):
            for q_len, k_len, causal in shapes:
                # Strided views, exactly as produced by the packed to_qkv projection.
                qkv = torch.randn(batch, k_len, 3 * 16 * 64, device='cuda', generator=gen)
                q_full, k_full, v_full = qkv.chunk(3, dim=-1)
                q = q_full[:, :q_len].reshape(batch, q_len, 16, 64).transpose(1, 2)
                k = k_full.reshape(batch, k_len, 16, 64).transpose(1, 2)
                v = v_full.reshape(batch, k_len, 16, 64).transpose(1, 2)
                mask = O.attention_mask(q_len, k_len, q.device, causal)
                reference = O.explicit_attention(q, k, v, mask)
                out = O.cuda_attend(q, k, v, causal)
                torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-6)
                self.assertTrue(out.transpose(1, 2).is_contiguous())

    @torch.inference_mode()
    def test_triton_rollout_matches_baseline(self):
        model = cuda_model()
        gen = torch.Generator(device='cuda').manual_seed(11)
        z = torch.randn(1, 1, 192, device='cuda', generator=gen)
        for layout in ('independent', 'late'):
            samples = torch.randn(1, 300, 5, 10, device='cuda', generator=gen)
            actions = O.TreePlan(layout, 'cuda').materialize(samples)
            reference = M.baseline_rollout(model, z, actions)
            out = O.rollout(model, z, actions, layout, raw_ops=True, use_cuda_attn=True)
            torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
