import torch


def mask_fraction(mask: torch.Tensor) -> float:
    return mask.float().mean().item()

