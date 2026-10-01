# TorchCFM coupling source

`optimal_transport.py` and `LICENSE` are unmodified copies from
https://github.com/atong01/conditional-flow-matching at commit
`7c653857de4c25b979a740ef8b6a5d5ddb09b6c1`.

Upstream file: `torchcfm/optimal_transport.py`. The included MIT license applies
to this source. The project calls `OTPlanSampler(method="exact").sample_plan`
with `replace=True`, its original default. For `[B, N, D]` inputs, each entire
cloud is one sample and the cost matrix is B by B. No within-cloud point
permutation is performed. The NumPy sampler is seeded with training seed + 1.

Reference: Tong et al., *Improving and generalizing flow-based generative
models with minibatch optimal transport*, https://arxiv.org/abs/2302.00482.

Only this coupling module is imported; the project's existing FM loss,
backbone, interpolation, time sampling and Euler evaluator are used.
