# PSF backbone + TG (single-GPU coupling comparison)

PSF is pinned as `third_party/PSF` at
`c74b39e1200513039cfb8d776505fb75da599e68`.
Source: <https://github.com/klightz/PSF>, MIT, copyright 2023 Lemeng Wu.
The tiny PVCNN2 block configuration in `psf_adapter.py` is copied from its
`train_flow.py`; all actual backbone layers, CUDA kernels and the ShapeNet
loader are imported from PSF. The full license remains in the submodule.

This is **PSF first-stage FM with a coupling comparison**, not a reproduction
of the full PSF reflow/distillation pipeline or the published NSOT benchmark.
The two YAML filenames are unchanged. Only the coupling differs between them.
The existing 2D scripts/configs are unchanged.

## What the patch changes

`patches/psf_windows.patch` only:

- uses MSVC flags on Windows and prints CUDA build logs;
- substitutes a portable function-name macro on MSVC;
- imports Open3D only when the optional shape-completion mask is invoked;
- sorts dataset filenames before upstream's fixed shuffle, for reproducible
  shape order across machines (no split membership or coordinates changed).

`prepare_psf.py` verifies the pinned commit, checks before applying, tolerates
repeated execution, and stops on conflicts. It never resets local changes.
The patched submodule appears modified in `git status`; that is expected.
**Commit the patch in this parent repository, not local edits inside PSF.**
If changing the patch later, reverse the old patch with `git apply --reverse`
after checking it, update the parent checkout, and prepare again. Do not reset
the submodule to discard changes. Preparation never downloads packages.

## Lab PC: fetch and prepare

In the existing project checkout:

```bat
git pull
git submodule sync --recursive
git submodule update --init --recursive
conda activate local_coupling
python -m pip install -r requirements.txt
python -m pip install ninja
python prepare_psf.py
```

For a fresh clone use `git clone --recurse-submodules <your-project-url>`.
Do **not** install PSF's old Torch/CUDA versions over the working environment.
No additional Open3D, PyTorch3D or PyTorchEMD installation is needed for these
entry points. POT's `ot.emd` is still used for exact **training coupling**;
it is unrelated to the excluded CUDA **EMD evaluation** extension.

The following commands are for **CMD**, not PowerShell. If already in the
working x64 developer CMD, do not open a nested shell. Otherwise open CMD and:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.44
conda activate local_coupling
set TORCH_CUDA_ARCH_LIST=
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
where cl
where nvcc
python psf_adapter.py
python psf_adapter.py --coupling target_guided_exact_optimized
```

Run from this repository root. Clearing `TORCH_CUDA_ARCH_LIST` lets PyTorch
detect the visible GPU. If setting it manually, RTX A6000 uses `8.6`, whereas
the previously reported RTX 6000 Ada uses `8.9`; do not confuse these devices.
These tests need no data and actually run forward, backward and an optimizer
step on the original PVCNN kernels. Require `PSF_FORWARD_BACKWARD_OK` for both
methods before training. The first call compiles to `.torch_extensions/`.
The earlier empty CUDA-kernel build probe does NOT guarantee these tests pass.
Further compiler/API errors must be diagnosed from their actual logs.

## Data and training

Download ShapeNet PC15k using the data link in the
[PSF README](https://github.com/klightz/PSF#data). Data is not included in Git.
Expected chair layout (each .npy contains 15,000 x 3 coordinates):

```text
data/ShapeNetCore.v2.PC15k/03001627/train/*.npy
data/ShapeNetCore.v2.PC15k/03001627/val/*.npy
data/ShapeNetCore.v2.PC15k/03001627/test/*.npy
```

The alternative archive layout `03001627/_/train/*.npy` (likewise `val` and
`test`) is also detected automatically. Keep `data.root` / `--dataroot` pointing
to `ShapeNetCore.v2.PC15k`, not to the category or `_` folder. No data moves,
symlinks, re-extraction, or additional `prepare_psf.py` call are needed.
Both layouts use the original PSF loading, normalization and point subsampling;
category/shape IDs are identical. If both layouts contain .npy files for the same
category/split, loading stops rather than silently selecting or merging them.
`02691156` is airplane; `03001627` is chair. The configs still default to chair;
to train airplane, set `data.category: airplane` in BOTH existing YAML files.

Start with a short Independent run:

```bat
python train_3d.py psf_experiments/independent.yaml --steps 10
python train_3d.py psf_experiments/target_guided.yaml --steps 10
```

Check peak VRAM and seconds per optimizer update in both logs. The first updates
are not steady-state timing. There is no automatic training-budget or OOM fallback.
Then compare both full training configs:

```bat
python train_3d.py psf_experiments/independent.yaml
python train_3d.py psf_experiments/target_guided.yaml
```

Use `--dataroot "D:\datasets\ShapeNetCore.v2.PC15k"` if needed (also available
in eval). Both configs use chair, N=2048, K=8, seed=0, and:

| Setting | Value in both methods |
| --- | --- |
| Microbatch (shapes per forward/backward) | 16 |
| Gradient accumulation | 16 microbatches per optimizer update |
| Effective batch | 256 shapes |
| Optimizer updates | 600,000 |
| Optimizer | Adam, beta1=0.5, beta2=0.999, weight_decay=0 (PSF choice) |
| Learning rate | Constant 0.0002; no scheduler |
| EMA | None; evaluate raw weights |
| Precision | float32; no AMP |

Microbatch 16 is a conservative starting setting for a 48-GB-class single GPU,
NOT a measured memory-fit guarantee. If it does not fit, set batch_size=8 and
accumulation_steps=32 in BOTH YAMLs before starting new runs. If increasing it,
keep their product 256 and use the same microbatch in both methods. Accumulation
does not save total computation. One `--steps 10` run still processes 2,560 shapes.
600,000 updates is a substantial single-GPU budget; inspect the measured cost
before committing to a long run. The same update/sample budget is not the same
wall-clock budget, so report training time alongside quality.

The provided TG uses the existing exact optimized implementation: the same POT
objective/constraints with batched transfers, not an approximate solver.

Both use stock PVCNN2, uniform t, linear paths, MSE, and t*999 embeddings.
Coupling only permutes target points during training. The internal random pairing
uses a separate RNG, so it does not consume the baseline's noise/time RNG stream.
The microbatch MSE is divided by accumulation_steps before backward; the optimizer
is updated only once per effective batch. No scheduler, EMA, AMP, reflow,
distillation, or upstream distributed/NCCL runner. DataLoader workers=0 for Windows;
the resumable sampler shuffles each epoch and drops its final incomplete microbatch.
CUDA reductions are not guaranteed bitwise deterministic. All methods retain
the same architecture, source noise/time distribution and Euler sampler.

## Relation to NSOT

[NSOT Appendix B](https://arxiv.org/html/2502.12456v1#A2) reports roughly 600,000
iterations, total batch 256, Adam with initial LR 2e-4, LR decay 0.998 every 1,000
iterations, and EMA 0.9999 on four A100s. We match the reported update/effective
batch budget, but deliberately omit EMA and LR decay at the user's request.
Adam betas remain the PSF settings; do not label them as verified NSOT settings.
This supports a controlled Independent-vs-TG experiment, not identical numerical
reproduction of NSOT. Microbatch accumulation and four-GPU execution are not
claimed to be bitwise equivalent. Fixed LR also does not guarantee convergence;
inspect the intermediate checkpoints under a common evaluation protocol.

The upstream loader loads the selected split into RAM and samples **with
replacement** from each shape's first 10,000 points. This behavior is preserved.
N=8192 is supported by this loader, but N=15000 is deliberately rejected rather
than silently clamped to 10000. High-N TG time/VRAM must be measured separately.
NSOT uses a 100K-point superset; the current data source is therefore another
intentional difference, as is the current pilot evaluation protocol below.

## Checkpoints and resume

Runs: `runs/psf3d/<unique-run>/config.yaml`, `training.json`, `checkpoint.pt`.
`checkpoint.pt` is saved every 5,000 optimizer updates and at completion.
Separate `checkpoint_step_050000.pt`, `checkpoint_step_100000.pt`, etc. are kept
at 50k/100k/200k/400k/600k. These snapshots also contain optimizer and RNG state.
Resume into a NEW run directory (the original artifacts are preserved):

```bat
python train_3d.py psf_experiments/independent.yaml --resume "runs\psf3d\YOUR_RUN\checkpoint.pt"
```

`--steps` always means the TOTAL target optimizer updates, not additional updates.
For example, first train both methods with `--steps 50000`, then resume each with
the same YAML and no `--steps` to continue toward 600,000. Point sampling, shuffled
shape order/position, Python/NumPy/Torch/CUDA RNGs and the coupling RNG are restored.
Resume requires the same training settings, microbatch, data IDs/normalization,
and source/PSF code hashes. Only data root, total target steps and logging/save
intervals may change. Dataset contents must remain unchanged. Legacy v1 pilot
checkpoints are still evaluable but cannot be resumed without optimizer/RNG state.

Snapshots record normalization, shape IDs, PSF revision, patch hash, coupling,
steps, effective batch, number of shape presentations, losses, peak allocated VRAM,
training time and code hashes. The PSF tracked working diff is also
hashed, so extra local backbone/kernel edits are not silently treated as stock PSF.
Timing excludes initial JIT build and dataset load; includes the training loop
and prior checkpoint writes, but not the currently-being-written checkpoint.
On resume, the previous recorded training time is carried forward; initialization
and downtime between runs are excluded.

## Evaluation (no EMD)

Use the checkpoint path printed by training:

```bat
python eval_3d.py "runs\psf3d\YOUR_RUN\checkpoint.pt" --nfe 1 2 4 8 16 32 64 128 --samples 32 --batch-size 4
```

Same source noise and reference shapes are used across NFEs for the same eval
seed. Euler m steps = m model calls per generated batch. Saved training mean/std
normalizes the held-out shapes; no refitting on val/test. References use the first
10k points of **held-out shapes**, not training shapes. Each method must use the
same category, split, seed, N and sample count. Both use raw (non-EMA) weights;
this choice and the training budget are recorded in evaluation JSON.
No test-time coupling or refinement.

Outputs: `eval_results/psf3d/<run+hash>/nfe_*.json`, `.npy`, `.png`.
Coordinates and CD are in the training-normalized domain (mean/std in JSON).
CD = sum of two directional mean squared distances. MMD-CD = reference-to-generated
minimum matching distance; COV-CD = reference coverage; 1-NNA-CD = leave-one-out
two-sample accuracy (near 0.5 desirable, not monotonically lower-is-better).
No arbitrary one-to-one CD between unrelated generated/reference shapes is used.

The default **32 shapes is a quick functional check, not publication evaluation**.
Pairwise CD scales quadratically in the number of shapes and points; chunking
reduces peak memory, not computation. Full benchmark evaluation should later
match the reference paper's normalization, split, sample counts and metrics.

## CPU validation

With dependencies installed and the patch prepared:

```bat
python -m unittest discover -s tests -p test_psf_bridge.py -v
```

These check the upstream architecture configuration, adapter time/layout mapping,
training-stat reuse, CD metrics, patch idempotence, identical YAML budgets,
accumulated-vs-full-batch toy gradients, and exact CPU toy-training resume.
They do not certify CUDA
compilation, numerical equivalence of GPU kernels, or generation quality.
