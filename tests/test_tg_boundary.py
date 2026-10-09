"""Capacity, pairing-law and integration checks for offline coarse tables."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import prepare_tg
import tg_boundary
import tg_cache
import train
import train_horse


MODES = ("boundary_guided", "random_swap_control")


class BoundaryAlgorithmTests(unittest.TestCase):
    @staticmethod
    def cloud():
        # Two close source columns cross a coarse boundary. Broad target
        # patches are adjacent according to the target geometry gate.
        axis = np.linspace(-1., 1., 16)
        source = np.column_stack((np.repeat([-.04, .04], 16), np.tile(axis, 2))).astype(np.float32)
        target = np.column_stack((np.repeat([-.4, .4], 16), np.tile(.6 * axis, 2))).astype(np.float32)
        labels = np.repeat([0, 1], 16).astype(np.int32)
        return source, target, labels, labels.copy()

    @staticmethod
    def options(**overrides):
        return tg_boundary.settings({
            "num_tables": 4, "source_neighbors": 4, "max_swaps": 4,
            "destination_budget": 1., "max_cost_increase": .5,
            "cost_match_tolerance": .1, **overrides,
        })

    def test_guidance_is_nontrivial_capacity_preserving_and_cost_matched(self):
        source, target, source_labels, target_labels = self.cloud()
        options = self.options()
        guided, control, stats = tg_boundary.build_tables(
            source, target, source_labels, target_labels, 2, options, 9)
        self.assertEqual(guided.shape, (4, 32))
        self.assertEqual(guided.dtype, np.int32)
        self.assertEqual(control.dtype, np.int32)
        np.testing.assert_array_equal(guided[0], source_labels)
        np.testing.assert_array_equal(control[0], source_labels)
        self.assertGreater(sum(stats["swaps_per_table"]), 0)
        self.assertLess(stats["guided_score"], stats["baseline_score"])
        self.assertTrue(np.all(np.diff(stats["guided_score_trace"]) <= 1e-10))
        centroids = np.stack([target[target_labels == k].astype(np.float64).mean(0) for k in range(2)])
        edges, denominators = tg_boundary.source_graph(source, options["source_neighbors"])
        recomputed = tg_boundary.graph_score(
            source, centroids, guided, edges,
            denominators + options["distance_epsilon"], options["tau"])
        self.assertAlmostEqual(recomputed, stats["guided_score"], places=8)
        original_centroids = centroids[source_labels]
        baseline_cost = np.square(source - original_centroids).sum(1).mean()
        self.assertAlmostEqual(stats["baseline_centroid_cost"], baseline_cost, places=6)
        for name, tables in (("guided", guided), ("control", control)):
            for table, swaps in zip(tables, stats["swaps_per_table"]):
                np.testing.assert_array_equal(np.bincount(table, minlength=2), [16, 16])
                self.assertEqual(np.count_nonzero(table != source_labels), 2 * swaps)
                cost = np.square(source - centroids[table]).sum(1).mean()
                self.assertLessEqual(cost, baseline_cost * (1 + options["max_cost_increase"]) + 1e-6)
            destination_rms = np.sqrt(np.square(centroids[tables] - original_centroids[None]).sum(2).mean(0))
            self.assertLessEqual(destination_rms.max(), stats["destination_rms_limit"] + 1e-6)
            self.assertAlmostEqual(destination_rms.max(), stats[f"{name}_max_destination_rms"], places=6)
        guided_cost = np.array(stats["guided_centroid_cost"])
        control_cost = np.array(stats["control_centroid_cost"])
        self.assertTrue(np.all(np.abs(guided_cost - control_cost)
                               <= options["cost_match_tolerance"] * baseline_cost + 1e-6))
        self.assertLessEqual(stats["max_pair_cost_delta_difference_fraction"],
                             options["cost_match_tolerance"] + 1e-8)
        self.assertLessEqual(stats["max_table_cost_difference_fraction"],
                             options["cost_match_tolerance"] + 1e-8)
        json.dumps(stats, allow_nan=False)

    def test_zero_swap_or_destination_budget_has_explicit_safe_fallback(self):
        source, target, source_labels, target_labels = self.cloud()
        for override in ({"max_swaps": 0}, {"destination_budget": 0.}):
            with self.subTest(override=override):
                guided, control, stats = tg_boundary.build_tables(
                    source, target, source_labels, target_labels, 2, self.options(**override), 9)
                expected = np.broadcast_to(source_labels, guided.shape)
                np.testing.assert_array_equal(guided, expected)
                np.testing.assert_array_equal(control, expected)
                self.assertEqual(sum(stats["swaps_per_table"]), 0)
                self.assertAlmostEqual(stats["guided_score"], stats["baseline_score"])

    def test_build_is_deterministic_input_preserving_and_rng_isolated(self):
        args = self.cloud()
        before = tuple(value.copy() for value in args)
        np.random.seed(74)
        torch.manual_seed(75)
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        first = tg_boundary.build_tables(*args, 2, self.options(), 9)
        repeated = tg_boundary.build_tables(*args, 2, self.options(), 9)
        for value, expected in zip(args, before):
            np.testing.assert_array_equal(value, expected)
        for value, expected in zip(first[:2], repeated[:2]):
            np.testing.assert_array_equal(value, expected)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)

    def test_boundary_options_reject_unknown_nonfinite_and_invalid_ranges(self):
        for options in ({"unknown": 1}, {"num_tables": 0}, {"source_neighbors": True},
                        {"tau": float("nan")}, {"destination_budget": -1.},
                        {"cost_match_tolerance": float("inf")}, {"max_swaps": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                tg_boundary.settings(options)

    def test_incremental_scores_equal_full_graph_change_including_shared_edge(self):
        source, target, source_labels, target_labels = self.cloud()
        source, target = source.astype(np.float64), target.astype(np.float64)
        centroids = np.stack([target[target_labels == k].mean(0) for k in range(2)])
        options = self.options()
        tables = np.repeat(source_labels[None], options["num_tables"], axis=0)
        edges, distances = tg_boundary.source_graph(source, options["source_neighbors"])
        denominators = distances + options["distance_epsilon"]
        candidates = edges[source_labels[edges[:, 0]] != source_labels[edges[:, 1]]][:8]
        self.assertGreater(len(candidates), 0)
        changes = (centroids[source_labels[candidates[:, 1]]]
                   - centroids[source_labels[candidates[:, 0]]]) / len(tables)
        velocity = centroids[source_labels] - source
        deltas = tg_boundary._score_deltas(
            velocity, changes, candidates, edges, denominators, options["tau"],
            tg_boundary._incident_edges(edges, len(source)))
        before = tg_boundary.graph_score(
            source, centroids, tables, edges, denominators, options["tau"])
        for candidate, delta in zip(candidates, deltas):
            changed = tables.copy()
            i, j = candidate
            changed[1, i], changed[1, j] = changed[1, j], changed[1, i]
            after = tg_boundary.graph_score(
                source, centroids, changed, edges, denominators, options["tau"])
            self.assertAlmostEqual(delta, after - before, places=10)

    def test_far_narrow_target_patches_are_not_mixed_by_source_proximity(self):
        source, target, source_labels, target_labels = self.cloud()
        target[:, 0] = np.repeat([-1., 1.], 16)
        target[:, 1] *= 1e-4
        guided, control, stats = tg_boundary.build_tables(
            source, target, source_labels, target_labels, 2,
            self.options(destination_budget=10., max_cost_increase=1.), 9)
        self.assertEqual(stats["allowed_target_patch_pairs"], [])
        self.assertEqual(stats["eligible_source_edges"], 0)
        self.assertTrue(stats["no_feasible_change"])
        for tables in (guided, control):
            np.testing.assert_array_equal(tables, np.broadcast_to(source_labels, tables.shape))

    def test_degenerate_single_patch_is_finite_and_invalid_inputs_rejected(self):
        source = np.array([[0., 0.]], dtype=np.float32)
        target = source.copy()
        labels = np.zeros(1, dtype=np.int32)
        guided, control, stats = tg_boundary.build_tables(source, target, labels, labels, 1, seed=0)
        self.assertTrue(stats["no_feasible_change"])
        self.assertEqual(stats["baseline_score"], 0.)
        self.assertEqual(stats["guided_score"], 0.)
        np.testing.assert_array_equal(guided, control)
        json.dumps(stats, allow_nan=False)
        source, target, labels, target_labels = self.cloud()
        changed_labels = target_labels.copy()
        changed_labels[0] = 1
        with self.assertRaisesRegex(ValueError, "capacities"):
            tg_boundary.build_tables(source, target, labels, changed_labels, 2)
        bad_source = source.copy()
        bad_source[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            tg_boundary.build_tables(bad_source, target, labels, target_labels, 2)

    def test_streaming_summary_counts_only_present_variable_length_trace_entries(self):
        summary = {"clouds": 0, "noop_clouds": 0, "metrics": {}}
        tg_cache._accumulate_boundary(summary, {
            "accepted_swaps": 0, "guided_score_trace": [5.], "flag": True,
        })
        tg_cache._accumulate_boundary(summary, {
            "accepted_swaps": 2, "guided_score_trace": [7., 4., 1.], "flag": False,
        })
        self.assertEqual(summary["clouds"], 2)
        self.assertEqual(summary["noop_clouds"], 1)
        self.assertNotIn("flag", summary["metrics"])
        initial = summary["metrics"]["guided_score_trace.0"]
        self.assertEqual(initial, {"count": 2, "sum": 12., "min": 5., "max": 7.})
        later = summary["metrics"]["guided_score_trace.1"]
        self.assertEqual(later["count"], 1)
        self.assertEqual(later["sum"] / later["count"], 4.)
        with self.assertRaises(FloatingPointError):
            tg_cache._accumulate_boundary(summary, {"accepted_swaps": 0, "bad": float("nan")})


class BoundaryCacheTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    @staticmethod
    def config(mode="boundary_guided", dataset="checkerboard", sampling="bank"):
        result = {
            "seed": 0, "device": "cpu", "dtype": "float32",
            "coupling": "target_guided_cached", "num_regions": 2,
            "tg_cache": {
                "path": f"{dataset}_{mode}_{sampling}", "sampling": sampling,
                "num_clouds": 8, "seed": 5, "prepare_batch_size": 4,
                "num_workers": 0,
            },
            "data": {"batch_size": 2, "n_points": 16},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2,
                      "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001,
                         "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_{mode}.pt",
        }
        if mode != "baseline":
            result["tg_cache"]["coarse_mode"] = mode
            result["tg_cache"]["boundary"] = {
                "num_tables": 3, "source_neighbors": 4, "max_swaps": 4,
            }
        if dataset == "checkerboard":
            result["data"]["grid_size"] = 4
        return result

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_default_v1_and_base_cloud_hashes_unchanged_across_variants(self):
        metadata = {}
        for mode in ("baseline", *MODES):
            config = self.config(mode)
            _, metadata[mode] = self.prepare(config)
        self.assertEqual(metadata["baseline"]["format_version"], 1)
        self.assertNotIn("coarse_tables", metadata["baseline"]["array_sha256"])
        for mode in MODES:
            meta = metadata[mode]
            self.assertEqual(meta["format_version"], 3)
            self.assertEqual(meta["coarse_mode"], mode)
            self.assertIn("coarse_tables", meta["array_sha256"])
            for name in ("source", "target", "source_labels", "target_labels", "capacities"):
                self.assertEqual(meta["array_sha256"][name],
                                 metadata["baseline"]["array_sha256"][name], name)
        self.assertEqual(metadata[MODES[0]]["boundary_summary"],
                         metadata[MODES[1]]["boundary_summary"])
        for mode in MODES:
            self.assertEqual(metadata[mode]["boundary_summary"]["clouds"], 8)
            self.assertIn("tg_boundary.py", metadata[mode]["source_sha256"])
            self.assertGreaterEqual(metadata[mode]["coarse_guidance_seconds"], 0.)
            self.assertGreaterEqual(metadata[mode]["precompute_seconds"],
                                    metadata[mode]["coarse_guidance_seconds"])

    def test_cached_tables_preserve_capacities_and_complete_target_multisets(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                config = self.config(mode)
                path, meta = self.prepare(config)
                tables = np.load(path / "coarse_tables.npy")
                capacities = np.load(path / "capacities.npy")
                source = np.load(path / "source.npy")
                target = np.load(path / "target.npy")
                target_labels = np.load(path / "target_labels.npy")
                self.assertEqual(tables.shape, (8, 3, 16))
                self.assertEqual(tables.dtype, np.int32)
                for index, cloud in enumerate(tables):
                    for table in cloud:
                        np.testing.assert_array_equal(
                            np.bincount(table, minlength=2), capacities[index])
                sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
                try:
                    rng = np.random.default_rng(19)
                    for index in range(8):
                        for _ in range(6):
                            paired_source, paired_target = sampler.dataset.draw(index, rng)
                            np.testing.assert_array_equal(paired_source.numpy(), source[index])
                            self.assertEqual(sorted(map(tuple, paired_target.numpy())),
                                             sorted(map(tuple, target[index])))
                    # Every table remains compatible with a fresh random fine bijection.
                    for table in tables[0]:
                        permutation = tg_cache.random_fine_permutation(
                            table, target_labels[0], 2, rng)
                        np.testing.assert_array_equal(target_labels[0][permutation], table)
                finally:
                    sampler.close()
                self.assertIn("fresh uniform random", meta["fine_pairing"])

    def test_preparation_is_deterministic_and_leaves_global_rng_untouched(self):
        torch.manual_seed(171)
        np.random.seed(172)
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        for mode in MODES:
            config = self.config(mode)
            _, first = self.prepare(config)
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
            after = np.random.get_state()
            self.assertEqual(after[0], numpy_state[0])
            np.testing.assert_array_equal(after[1], numpy_state[1])
            self.assertEqual(after[2:], numpy_state[2:])
            repeated = copy.deepcopy(config)
            repeated["tg_cache"]["path"] += "_repeat"
            _, second = self.prepare(repeated)
            self.assertEqual(first["array_sha256"], second["array_sha256"])

    def test_actual_n256_k8_both_datasets_table_scores_and_guards(self):
        for dataset in ("checkerboard", "horse"):
            for mode in MODES:
                with self.subTest(dataset=dataset, mode=mode):
                    config = self.config(mode, dataset)
                    config["num_regions"] = 8
                    config["data"]["n_points"] = 256
                    config["tg_cache"]["num_clouds"] = 2
                    path, meta = self.prepare(config, dataset)
                    source = np.load(path / "source.npy").astype(np.float64)
                    target = np.load(path / "target.npy").astype(np.float64)
                    source_labels = np.load(path / "source_labels.npy")
                    target_labels = np.load(path / "target_labels.npy")
                    tables = np.load(path / "coarse_tables.npy")
                    self.assertEqual(tables.shape, (2, 3, 256))
                    opt = meta["boundary_options"]
                    for index in range(2):
                        centroids = np.stack([target[index, target_labels[index] == k].mean(0)
                                              for k in range(8)])
                        original = source_labels[index]
                        baseline = np.broadcast_to(original, tables[index].shape)
                        np.testing.assert_array_equal(tables[index, 0], original)
                        for table in tables[index]:
                            np.testing.assert_array_equal(np.bincount(table, minlength=8), np.full(8, 32))
                        cost = np.square(source[index] - centroids[tables[index]]).sum(2).mean(1)
                        base_cost = np.square(source[index] - centroids[original]).sum(1).mean()
                        self.assertTrue(np.all(cost <= base_cost * (1 + opt["max_cost_increase"]) + 1e-10))
                        drift = np.square(centroids[tables[index]] - centroids[original][None]).sum(2).mean(0)
                        target_rms_squared = np.square(target[index] - target[index].mean(0)).sum(1).mean()
                        self.assertLessEqual(drift.max(), opt["destination_budget"] ** 2 * target_rms_squared + 1e-10)
                        if mode == "boundary_guided":
                            edges, distances = tg_boundary.source_graph(source[index], opt["source_neighbors"])
                            score_args = (edges, distances + opt["distance_epsilon"], opt["tau"])
                            self.assertLessEqual(
                                tg_boundary.graph_score(source[index], centroids, tables[index], *score_args),
                                tg_boundary.graph_score(source[index], centroids, baseline, *score_args) + 1e-10)

    def test_sampler_draws_hard_tables_and_resamples_fine_pairing_within_each(self):
        # A direct in-memory dataset isolates the sampling law from acceptance
        # rates: all rows are deliberately different capacity-preserving tables.
        labels = np.repeat([0, 1], 8).astype(np.int32)
        tables = np.repeat(labels[None], 3, axis=0)
        tables[1, [0, 8]] = tables[1, [8, 0]]
        tables[2, [1, 9]] = tables[2, [9, 1]]
        source = np.arange(32, dtype=np.float32).reshape(1, 16, 2)
        target = source + 100
        dataset = tg_cache._CloudDataset("unused", {
            "num_clouds": 1, "num_regions": 2,
            "boundary_options": {"num_tables": 3},
        }, 7)
        dataset.arrays = {
            "source": source, "target": target, "source_labels": labels[None],
            "target_labels": labels[None], "coarse_tables": tables[None],
        }
        observed, draws = [], {tuple(row): set() for row in tables}
        original_fine = tg_cache.random_fine_permutation

        def record(selected, target_ids, k, rng):
            observed.append(tuple(selected))
            return original_fine(selected, target_ids, k, rng)

        with patch.object(tg_cache, "random_fine_permutation", side_effect=record):
            rng = np.random.default_rng(13)
            for _ in range(100):
                paired_source, paired_target = dataset.draw(0, rng)
                np.testing.assert_array_equal(paired_source.numpy(), source[0])
                selected = observed[-1]
                draws[selected].add(tuple(paired_target.flatten().tolist()))
                # Target row j is identifiable by its first coordinate.
                indices = ((paired_target[:, 0].numpy() - 100) // 2).astype(int)
                np.testing.assert_array_equal(labels[indices], selected)
        self.assertEqual(set(observed), set(draws))
        self.assertTrue(all(len(permutations) > 1 for permutations in draws.values()))
        dataset.arrays = None

    def test_online_sampling_has_no_solver_or_guidance_and_fine_pairs_are_fresh(self):
        for mode in MODES:
            config = self.config(mode)
            self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
            try:
                rng, seen = np.random.default_rng(41), set()
                with patch.object(tg_boundary, "build_tables", side_effect=AssertionError("online guidance")), \
                        patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")), \
                        patch.object(coupling.ot, "sinkhorn", side_effect=AssertionError("online Sinkhorn")):
                    for _ in range(30):
                        source, target = sampler.dataset.draw(0, rng)
                        seen.add(tuple(target.flatten().tolist()))
                    left = sampler.sample(3, generator=torch.Generator().manual_seed(3))
                    right = sampler.sample(3, generator=torch.Generator().manual_seed(3))
                self.assertGreater(len(seen), 1)
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            finally:
                sampler.close()

    def test_bank_and_stream_preserve_law_and_stream_is_single_use(self):
        for mode in MODES:
            config = self.config(mode, sampling="stream")
            config["tg_cache"]["num_clouds"] = None
            path, meta = self.prepare(config)
            self.assertEqual(meta["num_clouds"], 4)
            expected = torch.from_numpy(np.load(path / "source.npy"))
            sampler = tg_cache.TGCachedPairSampler(
                config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                with patch.object(tg_boundary, "build_tables", side_effect=AssertionError("online guidance")):
                    first, _ = sampler.sample(2)
                    second, _ = sampler.sample(2)
                torch.testing.assert_close(torch.cat((first, second)), expected, rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, "exhausted"):
                    sampler.sample(2)
            finally:
                sampler.close()
            longer = copy.deepcopy(config)
            longer["training"]["num_steps"] = 3
            with self.assertRaisesRegex(ValueError, "exceed"):
                tg_cache.load_cache(longer, "checkerboard")

    def test_configuration_and_table_integrity_fail_loudly(self):
        bad_options = (
            ("coarse_mode", "unsupported"),
            ("boundary", {"unknown": True}),
            ("boundary", {"num_tables": 0}),
            ("fine_pairing", "exact"),
            ("fine_mode", "exact"),
        )
        for key, value in bad_options:
            config = self.config()
            config["tg_cache"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                tg_cache.prepare(config, "checkerboard")
            self.assertFalse(Path(config["tg_cache"]["path"]).exists())
        config = self.config("baseline")
        config["tg_cache"]["boundary"] = {}
        with self.assertRaisesRegex(ValueError, "non-baseline"):
            tg_cache.prepare(config, "checkerboard")
        self.assertFalse(Path(config["tg_cache"]["path"]).exists())
        config = self.config()
        path, meta = self.prepare(config)
        changed = copy.deepcopy(config)
        changed["tg_cache"]["coarse_mode"] = "random_swap_control"
        with self.assertRaises(ValueError):
            tg_cache.load_cache(changed, "checkerboard")
        changed = copy.deepcopy(config)
        changed["tg_cache"]["boundary"]["max_swaps"] = 2
        with self.assertRaises(ValueError):
            tg_cache.load_cache(changed, "checkerboard")
        tables = np.load(path / "coarse_tables.npy")
        tables[0, 0, 0] = 1 - tables[0, 0, 0]
        np.save(path / "coarse_tables.npy", tables)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            tg_cache.load_cache(config, "checkerboard")

    def test_stream_cli_derives_trainable_config_without_touching_input(self):
        for mode in MODES:
            config = self.config(mode)
            original = Path(f"{mode}.yaml")
            original.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = original.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                path, meta = prepare_tg.main(original, "checkerboard", sampling="stream")
            self.assertEqual(before, original.read_bytes())
            derived = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(derived["tg_cache"]["coarse_mode"], mode)
            self.assertEqual(derived["tg_cache"]["boundary"], config["tg_cache"]["boundary"])
            self.assertEqual(derived["tg_cache"]["sampling"], "stream")
            self.assertEqual(meta["num_clouds"], 4)
            tg_cache.load_cache(derived, "checkerboard")

    def test_experiment_yamls_only_change_cache_coupling_and_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", ""), ("horse_experiments", "horse_")):
            original = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_k8_n256_seed0.yaml").read_text())
            variants = []
            for mode in MODES:
                config = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_{mode}_k8_n256_seed0.yaml").read_text())
                variants.append(config)
                self.assertEqual({k: v for k, v in config.items() if k not in ("tg_cache", "checkpoint")},
                                 {k: v for k, v in original.items() if k not in ("tg_cache", "checkpoint")})
                self.assertEqual(config["tg_cache"]["coarse_mode"], mode)
                self.assertEqual(config["tg_cache"]["sampling"], "bank")
                self.assertNotEqual(config["tg_cache"]["path"], original["tg_cache"]["path"])
                tg_cache.settings(config)
            self.assertEqual(variants[0]["tg_cache"]["boundary"], variants[1]["tg_cache"]["boundary"])

    def test_windows_spawn_workers_both_modes_and_sampling_schemes(self):
        for mode, sampling in (("boundary_guided", "bank"), ("random_swap_control", "stream")):
            config = self.config(mode, sampling=sampling)
            config["tg_cache"]["num_workers"] = 1
            self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(
                config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                for _ in range(2):
                    source, target = sampler.sample(2)
                    self.assertEqual(source.shape, (2, 16, 2))
                    self.assertTrue(torch.isfinite(target).all())
            finally:
                sampler.close()

    def test_tiny_train_load_eval_both_datasets_modes_no_cache_at_inference(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                for mode in MODES:
                    with self.subTest(dataset=dataset, mode=mode):
                        config = self.config(mode, dataset)
                        path = Path(f"{dataset}_{mode}.yaml")
                        path.write_text(yaml.safe_dump(config), encoding="utf-8")
                        prepare_tg.main(path, dataset)
                        with patch.object(tg_boundary, "build_tables", side_effect=AssertionError("online guidance")), \
                                patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")):
                            run = trainer(path, steps=1)
                        _, trained, _, metadata = load_model(run / "config.yaml", model_class, dataset)
                        self.assertIn(mode, run.name)
                        self.assertTrue(metadata["training_config_verified"])
                        self.assertIn("cache_sha256", trained["tg_cache"])
                        self.assertEqual(trained["tg_cache"]["coarse_mode"], mode)
                        self.assertIn("precompute_seconds", metadata["coupling_details"])
                        self.assertEqual(metadata["coupling_details"]["coarse_mode"], mode)
                        self.assertIn("cached_coarse_tables_sha256", metadata["coupling_details"])
                        with patch.object(tg_cache, "load_cache", side_effect=AssertionError("inference cache")):
                            if dataset == "checkerboard":
                                output = evaluator(run / "config.yaml", 2, render=False)
                            else:
                                with patch.object(eval_horse, "render_comparison"), \
                                        patch("horse_regions.HorseRegions.render"):
                                    output = evaluator(run / "config.yaml", 2)
                        result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                        self.assertEqual(result["coupling_details"], metadata["coupling_details"])
                        self.assertTrue(np.isfinite(result["chamfer"]))


if __name__ == "__main__":
    unittest.main()
