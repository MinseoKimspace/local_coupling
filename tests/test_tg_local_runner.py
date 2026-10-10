"""Runner contracts: exact saved paths, resume behavior and preflight isolation."""

import json
from pathlib import Path

import pytest
import yaml

import run_tg_local as runner


def test_real_variant_configs_preserve_baseline_training_contract():
    root = Path(__file__).resolve().parents[1]
    for dataset in ("checkerboard", "horse"):
        prefix = "horse_" if dataset == "horse" else ""
        folder = root / f"{dataset}_experiments"
        baseline = yaml.safe_load((folder / f"{prefix}target_guided_cached_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
        paths = {baseline["tg_cache"]["path"]}
        for mode, subpatches in (("path_affine", 1), ("path_affine_subpatch", 2)):
            config = yaml.safe_load((folder / f"{prefix}target_guided_cached_{mode}_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
            for field in ("seed", "device", "dtype", "coupling", "num_regions", "data", "model", "training", "evaluation"):
                assert config[field] == baseline[field], (dataset, mode, field)
            for field in ("sampling", "num_clouds", "seed", "prepare_batch_size", "num_workers"):
                assert config["tg_cache"][field] == baseline["tg_cache"][field]
            assert config["tg_cache"]["coarse_mode"] == mode
            assert config["tg_cache"]["local"]["subpatches"] == subpatches
            assert config["tg_cache"]["path"] not in paths
            paths.add(config["tg_cache"]["path"])
            assert config["checkpoint"] != baseline["checkpoint"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for dataset in ("checkerboard", "horse"):
        for variant in runner.VARIANTS:
            path = runner.config_path(dataset, variant)
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump({
                "device": "cuda", "tg_cache": {"path": f"cache/{dataset}_{variant}", "sampling": "bank",
                    "num_clouds": 4096, "num_workers": 2},
            }), encoding="utf-8")
    return tmp_path


def test_dry_run_never_creates_manifest_or_imports_trainers(project, monkeypatch, capsys):
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: pytest.fail(f"Unexpected import {name}"))
    before = set(project.rglob("*"))
    assert runner.main(["--dataset", "all", "--dry-run"]) is None
    assert set(project.rglob("*")) == before
    output = capsys.readouterr().out
    assert "path_affine_subpatch" in output
    assert "train ->" in output


def test_eval_requires_explicit_manifest(project):
    with pytest.raises(ValueError, match="requires --manifest"):
        runner.main(["--stage", "eval"])


def test_prepare_train_eval_uses_returned_runs_and_resumes(project, monkeypatch):
    calls = []

    def prepare(job, **kwargs):
        calls.append(("prepare", job["variant"]))
        job.update(prepared=True, effective_config=job["input_config"], effective_sha256=job["input_sha256"])

    def train(job):
        calls.append(("train", job["variant"]))
        # The returned path is deliberately unrelated to timestamp/latest ordering.
        saved = project / "runs" / ("exact_" + job["variant"]) / "config.yaml"
        saved.parent.mkdir(parents=True)
        saved.write_text("checkpoint: exact.pt\n", encoding="utf-8")
        job.update(run_config=str(saved), run_config_sha256=runner._digest(saved))

    def evaluate(job, nfe):
        calls.append(("eval", job["run_config"], nfe))
        output = project / (job["variant"] + f"_{nfe}.json")
        output.write_text("{}", encoding="utf-8")
        return {"json": str(output), "json_sha256": runner._digest(output)}

    monkeypatch.setattr(runner, "_prepare", prepare)
    monkeypatch.setattr(runner, "_train", train)
    monkeypatch.setattr(runner, "_evaluate", evaluate)
    manifest = runner.main(["--dataset", "horse", "--nfe", "1", "4"])
    assert len([call for call in calls if call[0] == "train"]) == 3
    assert len([call for call in calls if call[0] == "eval"]) == 6
    assert all("exact_" in call[1] for call in calls if call[0] == "eval")
    calls.clear()
    runner.main(["--stage", "all", "--manifest", str(manifest), "--nfe", "1", "4"])
    assert calls == []
    runner.main(["--stage", "eval", "--manifest", str(manifest), "--nfe", "4", "--rerun-eval"])
    assert len(calls) == 3 and all(call[0] == "eval" for call in calls)


def test_preflight_cannot_train_and_does_not_modify_original(project, monkeypatch):
    original = {path: path.read_bytes() for path in project.rglob("*.yaml")}
    seen = []

    def prepare(job, **kwargs):
        config = yaml.safe_load(Path(job["input_config"]).read_text(encoding="utf-8"))
        seen.append(config)
        assert kwargs["preflight"] is True
        job.update(prepared=True, effective_config=job["input_config"], effective_sha256=job["input_sha256"])

    monkeypatch.setattr(runner, "_prepare", prepare)
    manifest = runner.main(["--dataset", "checkerboard", "--stage", "preflight", "--clouds", "8"])
    assert len(seen) == 3
    assert all(config["device"] == "cpu" and config["tg_cache"]["num_clouds"] == 8 for config in seen)
    assert all(Path(config["tg_cache"]["path"]).is_relative_to(manifest.parent) for config in seen)
    assert all(path.read_bytes() == content for path, content in original.items())
    with pytest.raises(ValueError, match="preflight manifest cannot"):
        runner.main(["--stage", "train", "--manifest", str(manifest)])


def test_changed_snapshot_fails_before_training(project, monkeypatch):
    def prepare(job, **kwargs):
        job.update(prepared=True, effective_config=job["input_config"], effective_sha256=job["input_sha256"])

    monkeypatch.setattr(runner, "_prepare", prepare)
    manifest = runner.main(["--dataset", "horse", "--stage", "prepare"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(payload["jobs"][0]["input_config"]).write_text("changed: true\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_train", lambda job: pytest.fail("Training must not start"))
    with pytest.raises(ValueError, match="changed since"):
        runner.main(["--stage", "train", "--manifest", str(manifest)])


def test_all_run_configs_validated_before_any_eval(project, monkeypatch):
    def prepare(job, **kwargs):
        job.update(prepared=True, effective_config=job["input_config"], effective_sha256=job["input_sha256"])

    def train(job):
        saved = project / (job["variant"] + "_saved.yaml")
        saved.write_text("checkpoint: saved.pt\n", encoding="utf-8")
        job.update(run_config=str(saved), run_config_sha256=runner._digest(saved))

    monkeypatch.setattr(runner, "_prepare", prepare)
    monkeypatch.setattr(runner, "_train", train)
    manifest = runner.main(["--dataset", "horse", "--stage", "prepare"])
    runner.main(["--stage", "train", "--manifest", str(manifest)])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(payload["jobs"][-1]["run_config"]).unlink()
    monkeypatch.setattr(runner, "_evaluate", lambda job, nfe: pytest.fail("Evaluation must not start"))
    with pytest.raises(FileNotFoundError, match="Saved run config is missing"):
        runner.main(["--stage", "eval", "--manifest", str(manifest)])


def test_preflight_report_shows_heldout_warning_without_full_metadata():
    job = {"dataset": "horse", "variant": "path_affine", "cache_path": "cache", "metadata_path": "metadata.json"}
    metadata = {"num_clouds": 8, "precompute_seconds": 6.4, "local_guidance_seconds": 6.,
                "large_array": list(range(1000)), "local_summary": {"noop_clouds": 6, "metrics": {
                    "baseline_score": {"mean": .7}, "guided_score": {"mean": .69},
                    "heldout_baseline_score": {"mean": .7}, "heldout_guided_score": {"mean": .701},
                    "min_scored_fraction": {"mean": .95, "min": .9}, "accepted_swaps": {"mean": .5},
                }}}
    report = runner.preflight_report(job, metadata)
    assert report["coarse_noop_clouds"] == 6
    assert report["min_scored_fraction"] == .9
    assert report["accepted_swaps_mean"] == .5
    assert "no evidence" in report["warning"]
    assert "large_array" not in report and "local_summary" not in report
    metadata["local_summary"]["metrics"]["heldout_guided_score"]["mean"] = .699
    assert runner.preflight_report(job, metadata)["warning"] is None
