import contextlib
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
import diagnose
import eval as eval_checkerboard
import eval_horse
import train
import train_horse
from experiment import read_config
from source_randomization import randomization_settings, randomize_assignment
from summarize_results import collect_results


ROOT = Path(__file__).resolve().parents[1]
METHOD = "target_guided_randomized"


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_budget_and_capacities_per_cloud(self):
        costs = np.array([[[1, 1.01], [1.01, 1]], [[1, 3], [3, 1]]], dtype=np.float64)
        baseline = np.array([[0, 1], [0, 1]])
        diagnostics = {}
        result = randomize_assignment(costs, baseline, randomization_settings(),
            generator=torch.Generator().manual_seed(31), diagnostics=diagnostics)
        self.assertGreater(diagnostics["per_cloud"][0]["accepted_swaps"], 0)
        np.testing.assert_array_equal(result[1], baseline[1])
        for row, original, info in zip(result, baseline, diagnostics["per_cloud"]):
            np.testing.assert_array_equal(np.bincount(row), np.bincount(original))
            self.assertLessEqual(info["final_mean_cost"], info["allowed_mean_cost"] + info["numerical_tolerance"])
        np.testing.assert_array_equal(baseline, [[0, 1], [0, 1]])

    def test_zero_controls_do_not_draw_randomness_even_with_ties(self):
        for options in ({"relative_budget": 0}, {"proposal_sweeps": 0}):
            gen = torch.Generator().manual_seed(21)
            state = gen.get_state().clone()
            labels = np.array([[0, 0, 1, 1]])
            result = randomize_assignment(np.ones((1, 4, 2)), labels, randomization_settings(options), generator=gen)
            np.testing.assert_array_equal(result, labels)
            torch.testing.assert_close(gen.get_state(), state)

    def test_invalid_settings(self):
        for options in ({"relative_budget": -1}, {"relative_budget": float("nan")},
                        {"relative_budget": True}, {"proposal_sweeps": 1.5}, {"proposal_sweeps": -1},
                        {"extra": 1}, [], {"proposal_sweeps": True}):
            with self.assertRaises(ValueError):
                randomization_settings(options)
        with self.assertRaises(ValueError):
            coupling.coupling_permutation(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2),
                coupling="target_guided", num_regions=1, source_randomization={})

    def check_device(self, device):
        source, target = torch.randn(2, 17, 3, device=device), torch.rand(2, 17, 3, device=device)
        before_source, before_target = source.clone(), target.clone()
        for k in (1, 4, 17):
            local = lambda: torch.Generator(device=device).manual_seed(17)
            for data in (target, torch.zeros_like(target)):
                baseline = coupling.coupling_permutation(source, data, coupling="target_guided", num_regions=k, generator=local())
                actual = coupling.coupling_permutation(source, data, coupling=METHOD, num_regions=k,
                    source_randomization={"relative_budget": 0}, generator=local())
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
            diagnostics, gen = {}, local()
            cpu_state = torch.get_rng_state().clone()
            cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
            with patch.object(coupling.ot, "emd", wraps=coupling.ot.emd) as solve, \
                    patch.object(coupling, "pair_within_regions", wraps=coupling.pair_within_regions) as pair:
                actual = coupling.coupling_permutation(source, target, coupling=METHOD, num_regions=k,
                    source_randomization={"relative_budget": 0.05}, generator=gen,
                    assignment_generator=torch.Generator().manual_seed(53), coupling_diagnostics=diagnostics)
            self.assertEqual(solve.call_count, 4)  # B target + B source, not a third solve.
            self.assertEqual(pair.call_args.kwargs["local"], "random")
            _, _, source_labels, target_labels, _ = pair.call_args.args
            for a, b in zip(source_labels, target_labels):
                torch.testing.assert_close(torch.bincount(a, minlength=k), torch.bincount(b, minlength=k))
            torch.testing.assert_close(actual.sort(1).values.cpu(), torch.arange(17).expand(2, -1))
            torch.testing.assert_close(torch.get_rng_state(), cpu_state)
            if cuda_state is not None:
                torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_state)
            refgen = local()
            coupling.coupling_permutation(source, target, coupling="target_guided", num_regions=k, generator=refgen)
            torch.testing.assert_close(gen.get_state(), refgen.get_state())
            repeated = coupling.coupling_permutation(source, target, coupling=METHOD, num_regions=k,
                source_randomization={"relative_budget": 0.05}, generator=local(),
                assignment_generator=torch.Generator().manual_seed(53))
            torch.testing.assert_close(actual, repeated)
            self.assertLessEqual(diagnostics["max_relative_cost_increase"], 0.05 + 1e-12)
        torch.testing.assert_close(source, before_source)
        torch.testing.assert_close(target, before_target)

    def test_cpu_invariants_and_reproducibility(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_invariants_and_reproducibility(self):
        self.check_device("cuda")

    def test_configs_and_separate_budget_aggregation(self):
        paths = list((ROOT / "randomized_experiments").rglob("*.yaml"))
        self.assertEqual(len(paths), 18)
        checkpoints = []
        for path in paths:
            config = read_config(path)
            baseline_path = (ROOT / "horse_experiments/horse_target_guided_k8_n256_seed0.yaml"
                             if path.parent.name == "horse" else ROOT / "checkerboard_experiments/target_guided.yaml")
            baseline = read_config(baseline_path)
            for key in ("data", "model", "training", "evaluation", "num_regions", "dtype", "device"):
                self.assertEqual(config[key], baseline[key])
            checkpoints.append(config["checkpoint"])
        self.assertEqual(len(set(checkpoints)), 18)
        with tempfile.TemporaryDirectory() as tmp:
            for budget in (0.01, 0.05):
                for seed in (0, 1, 2):
                    record = {"dataset": "horse", "config": {"seed": seed, "coupling": METHOD,
                        "source_randomization": {"relative_budget": budget, "proposal_sweeps": 4}},
                        "training_config_verified": True, "checkpoint_sha256": f"{budget}_{seed}",
                        "evaluated_at": "2026-01-01", "euler_steps": 1, "chamfer": 1}
                    Path(tmp, f"{budget}_{seed}.json").write_text(json.dumps(record))
            groups, _ = collect_results(tmp, [0, 1, 2])
            self.assertEqual(len(groups), 2)

    def test_train_eval_and_diagnose_both_datasets(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(tmp)
                for dataset in ("checkerboard", "horse"):
                    config = read_config(ROOT / f"randomized_experiments/{dataset}/tg_budget01_k8_n256_seed0.yaml")
                    config["device"] = "cpu"
                    config["data"].update(batch_size=2, n_points=16)
                    config["evaluation"]["batch_size"] = 2
                    config["num_regions"] = 4
                    config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                    config["training"]["num_steps"] = 1
                    Path("config.yaml").write_text(yaml.safe_dump(config))
                    run = (train_horse if dataset == "horse" else train).main("config.yaml")
                    record = json.loads((run / "training.json").read_text())
                    self.assertEqual(len(record["coupling_diagnostics"]["records"]), 1)
                    artifact = (eval_horse if dataset == "horse" else eval_checkerboard).main(run / "config.yaml", 2)
                    self.assertTrue(json.loads(artifact.with_suffix(".json").read_text())["training_config_verified"])
                    with patch.object(diagnose, "NFES", (1, 2)), patch.object(diagnose, "TIMES", (0., 0.5, 1.)), \
                            patch.object(diagnose, "BINS", ((0., 1.),)), patch.object(diagnose, "render"):
                        output = diagnose.diagnose(run / "config.yaml", dataset, batches=1, batch_size=2, reference_nfe=2)
                    result = json.loads((output / "diagnostics.json").read_text())
                    self.assertEqual(len(result["coupling_diagnostics"]), 1)
                    frozen = yaml.safe_load((run / "config.yaml").read_text())
                    frozen["source_randomization"]["relative_budget"] = 0.05
                    (run / "bad.yaml").write_text(yaml.safe_dump(frozen))
                    with self.assertRaisesRegex(ValueError, "source_randomization"):
                        (eval_horse if dataset == "horse" else eval_checkerboard).main(run / "bad.yaml", 1)
            finally:
                os.chdir(previous)
