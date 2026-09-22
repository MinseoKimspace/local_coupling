import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from audit_target_patches import (audit, cell_ids, chord_outside, connectivity, neighbor_edges,
                                  sample_pairs, support_query)
from coupling import balanced_target_partition
from experiment import read_config
from train_horse import load_horse_mask, sample_horse


ROOT = Path(__file__).resolve().parents[1]


class PatchAuditTests(unittest.TestCase):
    def test_known_checkerboard_support_and_chords(self):
        points = np.array([[-.75, .75], [-.6, .6], [.25, .75], [2., 0.]])
        np.testing.assert_array_equal(cell_ids(points, 4), [0, 0, 2, -1])
        inside, spacing = support_query("checkerboard")
        exits = chord_outside(points, np.array([[0, 1], [0, 2]]), inside, spacing)
        self.assertEqual(exits[0], 0)
        self.assertGreater(exits[1], 0)

    def test_horse_inverse_matches_sampler(self):
        mask = load_horse_mask("cpu", torch.float64)
        points = sample_horse(mask, 1, 1000)[0].numpy()
        inside, _ = support_query("horse", mask=mask.numpy().astype(bool))
        self.assertTrue(inside(points).all())
        self.assertFalse(inside(np.array([[5., 5.]])).any())

    def test_graph_connectivity_not_just_full_cloud(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3]])
        result = connectivity(4, pairs, np.array([0, 1, 0, 1]), 2)
        self.assertEqual(result["global_components"], 1)
        self.assertEqual(result["fragmented_patch_fraction"], 1)
        self.assertEqual(result["mean_largest_component_fraction"], .5)
        self.assertEqual(result["patches"][0]["global_components_spanned"], 1)
        result = connectivity(4, np.array([[0, 1], [2, 3]]), np.array([0, 0, 1, 1]), 2)
        self.assertEqual(result["global_components"], 2)
        self.assertEqual(result["fragmented_patch_fraction"], 0)

    def test_graph_3d_coincident_points_and_singleton(self):
        points = np.zeros((4, 3))
        edges, lengths, scale = neighbor_edges(points, 3)
        self.assertEqual(len(edges), 6)
        self.assertEqual(scale, 0)
        self.assertTrue((lengths == 0).all())
        self.assertEqual(connectivity(4, edges, np.zeros(4, dtype=int), 1)["global_components"], 1)
        edges, _, _ = neighbor_edges(points[:1], 3)
        result = connectivity(1, edges, np.zeros(1, dtype=int), 1)
        self.assertEqual(result["isolated_point_fraction"], 1)
        self.assertEqual(result["mean_largest_component_fraction"], 1)

    def test_pair_sampling_no_self_pairs(self):
        rng = np.random.default_rng(4)
        self.assertEqual(len(sample_pairs(np.arange(3), 10, rng)), 3)
        pairs = sample_pairs(np.arange(100), 32, rng)
        self.assertEqual(len(pairs), 32)
        self.assertTrue((pairs[:, 0] != pairs[:, 1]).all())
        self.assertEqual(sample_pairs(np.arange(1), 10, rng).shape, (0, 2))

    def test_audit_json_and_figures_both_datasets(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for dataset in ("checkerboard", "horse"):
                config = read_config(ROOT / f"randomized_experiments/{dataset}/tg_baseline_k8_n256_seed0.yaml")
                config["device"] = "cpu"
                config["data"]["n_points"] = 32
                config["num_regions"] = 4
                path = Path(tmp, "config.yaml")
                path.write_text(yaml.safe_dump(config))
                output = audit(path, dataset, clouds=2, pairs_per_patch=16, output=tmp)
                result = json.loads((output / "patch_audit.json").read_text())
                for name in ("patches.png", "connectivity.png"):
                    self.assertGreater((output / name).stat().st_size, 1000)
                self.assertEqual(len(result["records"]), 2)
                self.assertEqual(result["records"][0]["balanced"]["capacities"], [8] * 4)
                points = torch.tensor(result["example"]["points"], dtype=torch.float32).unsqueeze(0)
                anchors, labels, _, _ = balanced_target_partition(points, 4)
                self.assertEqual(labels[0].tolist(), result["example"]["balanced"]["labels"])
                self.assertEqual(anchors[0].tolist(), result["example"]["anchors"])
                if dataset == "checkerboard":
                    self.assertEqual(sum(result["records"][0]["true_cell_counts"]), 32)
                    self.assertIn("point_weighted_cell_mixing_fraction", result["summary"]["balanced"])
                other = audit(path, dataset, clouds=1, pairs_per_patch=4, output=tmp)
                self.assertNotEqual(other, output)
