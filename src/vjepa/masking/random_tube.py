import torch


class RandomTubeMask:
    """Masks the same spatial patches across all temporal tubelets."""

    def __init__(self, ratio: float = 0.65):
        self.ratio = ratio

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None) -> torch.Tensor:
        t, h, w = grid
        spatial = h * w
        count = max(1, int(spatial * self.ratio))
        mask = torch.zeros(batch_size, t, spatial, dtype=torch.bool, device=device)
        noise = torch.rand(batch_size, spatial, device=device)
        selected = noise.argsort(dim=1)[:, :count]
        mask.scatter_(2, selected.unsqueeze(1).expand(-1, t, -1), True)
        return mask.flatten(1)

