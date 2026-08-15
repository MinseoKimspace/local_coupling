import torch


def checkerboard_centers(
    grid_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = torch.arange(grid_size, device=device)

    rows, cols = torch.meshgrid(
        indices,
        indices,
        indexing="ij",
    )

    active = (rows + cols) % 2 == 0
    cell_size = 2.0 / grid_size
    x = -1.0 + (cols[active].to(dtype) + 0.5) * cell_size
    y = 1.0 - (rows[active].to(dtype) + 0.5) * cell_size

    return torch.stack([x, y], dim=-1)


def sample_checkerboard(
    batch_size: int,
    n_points: int,
    device: torch.device | str,
    dtype: torch.dtype,
    grid_size: int = 4,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    base_centers = checkerboard_centers(
        grid_size=grid_size,
        device=device,
        dtype=dtype,
    )  # [K, 2]

    n_cells = base_centers.shape[0]
    cell_size = 2.0 / grid_size

    cell_index = torch.randint(
        low=0,
        high=n_cells,
        size=(batch_size, n_points),
        device=device,
    )  # [B, N]

    selected_centers = base_centers[cell_index]

    local_position = (
        torch.rand(
            batch_size,
            n_points,
            2,
            device=device,
            dtype=dtype,
        ) - 0.5
    ) * cell_size

    return selected_centers + local_position
