import torch


def farthest_point_sample(
    points: torch.Tensor,
    num_samples: int,
    *,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def balanced_partition(
    points: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError


def match_regions(
    source: torch.Tensor,
    target: torch.Tensor,
    source_regions: torch.Tensor,
    target_regions: torch.Tensor,
    *,
    num_regions: int,
) -> torch.Tensor:
    raise NotImplementedError


def independent_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    raise NotImplementedError


def regional_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    num_regions: int,
    random_fps_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    raise NotImplementedError
