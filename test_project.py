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


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class CudaTest(unittest.TestCase):
    def test_compiled_baseline_matches_eager_baseline(self):
        model = M.load_model('cuda')
        pair = M.image_pair(12)
        reference = M.request(model, *pair, 47)
        compiled = O.CompiledGraphBaseline(model).request(*pair, 47)
        for key in reference:
            torch.testing.assert_close(compiled[key], reference[key], rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
