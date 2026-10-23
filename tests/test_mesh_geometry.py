"""One shared entry: numerical contracts and real mesh-to-report integration."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_mesh_geometry as ev

NATIVE = importlib.util.find_spec("open3d") is not None
ROOT = Path(__file__).resolve().parents[1]


def square(z=0., x1=1.):
    return np.array([[0, 0, z], [x1, 0, z], [x1, 1, z], [0, 1, z]], float), np.array([[0, 1, 2], [0, 2, 3]])


def ply(path, vertices, triangles):
    # File fixture exercises the real Open3D reader, not a mocked mesh dictionary.
    text = ['ply', 'format ascii 1.0', f'element vertex {len(vertices)}',
            'property float x', 'property float y', 'property float z',
            f'element face {len(triangles)}', 'property list uchar int vertex_indices', 'end_header']
    text += [' '.join(map(str, p)) for p in vertices]
    text += ['3 ' + ' '.join(map(str, t)) for t in triangles]
    path.write_text('\n'.join(text) + '\n', encoding='ascii')


class NumericalTests(unittest.TestCase):
    def test_direction_strict_threshold_and_zero_f1(self):
        m = ev.summarize_distances(np.array([0, .01, .03, .1]), np.array([0, .02]), [.01, .02, .05])
        self.assertAlmostEqual(m['accuracy_mm'], 35.)
        self.assertAlmostEqual(m['completeness_mm'], 10.)
        self.assertAlmostEqual(m['surface_chamfer_mm'], 22.5)
        self.assertEqual(m['prf'][0]['precision'], .25)
        self.assertEqual(m['prf'][0]['recall'], .5)
        self.assertAlmostEqual(m['prf'][0]['f1'], 1/3)
        self.assertEqual(ev.summarize_distances([1], [1], [.01])['prf'][0]['f1'], 0.)

    def test_invalid_inputs_do_not_become_scores(self):
        for d in ([], [np.nan], [-.1], [[.1]]):
            with self.assertRaises(ValueError):
                ev.summarize_distances(d, [.1], [.05])
        v, t = square()
        for bad_v, bad_t in ((v * np.nan, t), (v, np.array([[0, 1, 9]])), (v, np.array([[0, 0, 0]]))):
            with self.assertRaises(ValueError):
                ev.prepare_mesh(bad_v, bad_t)
        for overrides in ({'thresholds_m': [0]}, {'unit_scale_to_m': 0}, {'sample_count': 0},
                          {'primary_threshold_m': .03}, {'seed': -1}, {'unexpected': 1}):
            config = json.loads((ROOT/'experiments/mesh_geometry/scannetpp_surface_v1.json').read_text())
            config.update(overrides)
            with self.assertRaises(ValueError):
                ev.validate_protocol(config)

    def test_uniform_area_sampler_and_local_reproducibility(self):
        # Two spatially separated triangles of areas .5 and 2: expected 20%/80%.
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]], float)
        mesh = ev.prepare_mesh(v, np.array([[0,1,2],[3,4,5],[0,0,0]]), chunk_size=1)
        self.assertEqual(mesh['removed_zero_area_faces'], 1)
        np.random.seed(31)
        before = np.random.get_state()
        points = ev.surface_samples(mesh, 20000, 0, chunk_size=73)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(points, ev.surface_samples(mesh, 20000, 0, chunk_size=73))
        np.testing.assert_array_equal(points, ev.surface_samples(mesh, 20000, 0, chunk_size=1001))
        self.assertLess(abs(np.mean(points[:,0] < 2) - .2), .015)
        small = points[points[:,0] < 2]
        self.assertTrue(np.all(small >= 0))
        self.assertTrue(np.all(small[:,:2].sum(axis=1) <= 1 + 1e-12))


@unittest.skipUnless(NATIVE, 'Open3D required: native tests did not execute')
class NativeTests(unittest.TestCase):
    def test_orchestration_uses_declared_seed_for_every_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'inputs'
            inputs.mkdir()
            gt, model = inputs/'gt.ply', inputs/'model.ply'
            for path in (gt, model):
                ply(path, *square())
            config = json.loads((ROOT/'experiments/mesh_geometry/scannetpp_surface_v1.json').read_text())
            config['sample_count'] = 64
            # Spy wraps the REAL sampler and REAL mesh producer/consumer chain.
            with patch.object(ev, 'surface_samples', wraps=ev.surface_samples) as sample:
                ev.evaluate_meshes(gt, {'baseline':model}, config, root/'eval', 'plane', {'baseline':'test'})
            self.assertEqual([call.args[2] for call in sample.call_args_list], [0, 0])

    def test_triangle_interior_and_translated_planes(self):
        mesh = ev.prepare_mesh(*square())
        d = ev.closest_triangle_distances(np.array([[.25,.4,0],[.25,.4,.04]]), mesh, chunk_size=1)
        np.testing.assert_allclose(d, [0, .04], atol=1e-7)
        points = ev.surface_samples(ev.prepare_mesh(*square(.03)), 1000, 2)
        np.testing.assert_allclose(ev.closest_triangle_distances(points, mesh), .03, atol=1e-7)

    def test_missing_surface_affects_completeness(self):
        gt, half = ev.prepare_mesh(*square()), ev.prepare_mesh(*square(x1=.5))
        p, q = ev.surface_samples(half, 10000, 0), ev.surface_samples(gt, 10000, 0)
        m = ev.summarize_distances(ev.closest_triangle_distances(p, gt), ev.closest_triangle_distances(q, half), [.05])
        self.assertLess(m['accuracy_mm'], .001)
        self.assertGreater(m['completeness_mm'], 100)
        self.assertGreater(m['prf'][0]['precision'], .99)
        self.assertLess(m['prf'][0]['recall'], .6)

    def test_real_tnt_scalar_producer_agrees(self):
        from scripts.tnt_eval.evaluation import get_f1_score_histo2
        a, b = np.array([0, .01, .03, .1]), np.array([0, .02])
        old = get_f1_score_histo2(.02, '', 3, a, b)
        new = ev.summarize_distances(a, b, [.02])['prf'][0]
        np.testing.assert_allclose(old[:3], [new[k] for k in ('precision','recall','f1')])

    def test_real_ply_cli_serializer_csv_units_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root/'inputs'
            inputs.mkdir()
            gt, base, candidate = [inputs/n for n in ('gt.ply','baseline.ply','candidate.ply')]
            # Millimetre inputs, explicit conversion; known 40mm -> 0mm improvement.
            for p, z in ((gt,0),(base,40),(candidate,0)):
                v,t = square(z=z, x1=1000)
                v[:,1] *= 1000
                ply(p,v,t)
            initial = {p: p.read_bytes() for p in (gt,base,candidate)}
            protocol = json.loads((ROOT/'experiments/mesh_geometry/scannetpp_surface_v1.json').read_text())
            protocol.update(sample_count=2000, unit_scale_to_m=.001)
            config = root/'protocol.json'
            config.write_text(json.dumps(protocol))
            out = root/'evaluation'
            cmd = [sys.executable, '-B', str(ROOT/'scripts/evaluate_mesh_geometry.py'),
                   '--config', str(config), '--scene', 'synthetic_plane', '--gt', str(gt),
                   '--mesh', f'baseline={base}', '--mesh', f'selector={candidate}',
                   '--model-commit', 'baseline=test-base', '--model-commit', 'selector=test-new',
                   '--output', str(out)]
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report = json.loads((out/'results.json').read_text())
            with (out/'metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),6)
            self.assertEqual([x['model'] for x in report['models']], ['baseline','selector'])
            self.assertAlmostEqual(report['models'][0]['metrics']['accuracy_mm'],40.,places=4)
            self.assertLess(report['models'][1]['metrics']['surface_chamfer_mm'], .001)
            self.assertLess(report['models'][1]['delta_from_baseline']['surface_chamfer_mm'], -39.99)
            self.assertEqual(len(report['gt_samples_sha256']),64)
            self.assertEqual(report['models'][0]['input']['sha256'],hashlib.sha256(initial[base]).hexdigest())
            self.assertEqual(json.loads((out/'resolved_config.json').read_text())['protocol'], protocol)
            self.assertIn('git_commit',json.loads((out/'run_identity.json').read_text()))
            saved = (out/'results.json').read_bytes()
            again = subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertEqual((out/'results.json').read_bytes(),saved)
            for p,content in initial.items(): self.assertEqual(p.read_bytes(),content)


if __name__ == '__main__':
    unittest.main()
