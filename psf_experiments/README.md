# PSF backbone + TG (Windows pilot)

PSF is pinned as `third_party/PSF` at
`c74b39e1200513039cfb8d776505fb75da599e68`.
Source: <https://github.com/klightz/PSF>, MIT, copyright 2023 Lemeng Wu.
The tiny PVCNN2 block configuration in `psf_adapter.py` is copied from its
`train_flow.py`; all actual backbone layers, CUDA kernels and the ShapeNet
loader are imported from PSF. The full license remains in the submodule.

This is **PSF first-stage FM with a coupling comparison**, not a reproduction
of the full PSF reflow/distillation pipeline or the published benchmark.
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
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
where cl
where nvcc
python psf_adapter.py
python psf_adapter.py --coupling target_guided_exact_optimized
```

Run from this repository root. `8.9` targets the lab RTX 6000 Ada, not every GPU.
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

Start with a short Independent run:

```bat
python train_3d.py psf_experiments/independent.yaml --steps 100
```

Then compare both full pilot configs:

```bat
python train_3d.py psf_experiments/independent.yaml
python train_3d.py psf_experiments/target_guided.yaml
```

Use `--dataroot "D:\datasets\ShapeNetCore.v2.PC15k"` if needed (also available
in eval). Configs: chair, N=2048, K=8, batch=8, seed=0, 10,000 optimizer steps.
This budget is a smoke/pilot choice, not a claim of convergence. Batch 8 is a
conservative starting point; increase both configs together after measuring VRAM.
The provided TG uses the existing exact optimized implementation: the same POT
objective/constraints with batched transfers, not an approximate solver.

Both use stock PVCNN2, uniform t, linear paths, MSE, and t*999 embeddings.
Coupling only permutes target points during training. The internal random pairing
uses a separate RNG, so it does not consume the baseline's noise/time RNG stream.
Adam follows PSF's beta1=0.5; LR decays by 0.998 per completed epoch. No EMA,
AMP, reflow or distillation; no upstream distributed/NCCL runner. DataLoader
workers=0 for Windows. CUDA reductions are not guaranteed bitwise deterministic.

The upstream loader loads the selected split into RAM and samples **with
replacement** from each shape's first 10,000 points. This behavior is preserved.
N=8192 is supported by this loader, but N=15000 is deliberately rejected rather
than silently clamped to 10000. High-N TG time/VRAM must be measured separately.

Runs: `runs/psf3d/<unique-run>/config.yaml`, `training.json`, `checkpoint.pt`.
Checkpoint saved every 1,000 steps and at completion. Only the latest weights
per run are retained; this minimal runner does not implement training resume.
Snapshots record normalization, shape IDs, PSF revision, patch hash, coupling,
steps, losses, training time and code hashes. The PSF tracked working diff is also
hashed, so extra local backbone/kernel edits are not silently treated as stock PSF.
Timing excludes initial JIT build
and dataset load; includes the training loop and prior checkpoint writes.

## Evaluation (no EMD)

Use the checkpoint path printed by training:

```bat
python eval_3d.py "runs\psf3d\YOUR_RUN\checkpoint.pt" --nfe 1 2 4 8 16 32 64 128 --samples 32 --batch-size 4
```

Same source noise and reference shapes are used across NFEs for the same eval
seed. Euler m steps = m model calls per generated batch. Saved training mean/std
normalizes the held-out shapes; no refitting on val/test. References use the first
10k points of **held-out shapes**, not training shapes. Each method must use the
same category, split, seed, N and sample count. No test-time coupling or refinement.

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
training-stat reuse, CD metrics and patch idempotence. They do not certify CUDA
compilation, numerical equivalence of GPU kernels, or generation quality.
