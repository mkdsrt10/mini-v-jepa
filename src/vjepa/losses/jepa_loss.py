import torch
import torch.nn.functional as F


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Feature-space regression with unit-normalized embeddings."""
    return F.mse_loss(F.normalize(predicted, dim=-1), F.normalize(target.detach(), dim=-1))

