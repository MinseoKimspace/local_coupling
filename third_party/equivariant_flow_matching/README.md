# Equivariant Flow Matching: permutation-only coupling

This is an adaptation of the author-provided supplementary implementation for
Leon Klein, Andreas Krämer, and Frank Noé, *Equivariant flow matching*
(NeurIPS 2023): https://arxiv.org/abs/2306.15030.

Source: user-provided archive
`11904_Equivariant_flow_matchin_Supplementary Material.zip`.

- Archive SHA-256: `21be6d364baa77f2d3f86c8bd337d1971ff1c1a579f1822d6204800742ca83d6`.
- Notebook: `supplementary_material/DW4_eq_OT_flow_matching.ipynb`.
- Cell: zero-based index 10.
- `original_coupling.txt` is the verbatim coupling block, starting at
  `# Resample x0, x1 according to transport matrix` and ending immediately
  before `x0 = x0.cuda()`. It is a non-executed reference.
- Reference block UTF-8 SHA-256: `1d73d94f9e07d799e4753f7168d5d3757cfee99b2cccf39c3f84b1f9a374df0e`.

The adapter preserves the notebook's point-Hungarian matching for every cloud
pair, normalization of the resulting cloud costs, exact POT cloud transport,
NumPy sampling with replacement, and the second point-Hungarian solve for each
selected cloud pair. NumPy's global random state controls cloud-plan sampling.

The authorized change removes both SVD alignments. Target coordinates retain
their original orientation, so the fixed-orientation checkerboard and horse
distributions remain the comparison targets. This is **permutation-only EFM
coupling**, not the unmodified rotationally equivariant algorithm or a
reproduction of its full model.

Interface changes accept and return `[B, N, D]` tensors, move solver inputs to
CPU, and restore the input device and dtype. Low-precision inputs use float32
for CPU distance calculations. Inputs are not modified. All-zero cloud costs
skip division by zero. No centering, mean-free prior, interpolation noise,
backbone, or inference changes are imported.

The candidate-pair stage solves B squared point-assignment problems, followed
by B selected-pair solves. With B=1 its permutation objective coincides with
the project's `global_ot` objective, apart from solver tie-breaking and
floating-point details.

The supplied archive contains no LICENSE file or explicit license grant.
Attribution here does not assign a license to the supplementary source; its
rights remain with the authors.
