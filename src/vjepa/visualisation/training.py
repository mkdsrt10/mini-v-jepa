"""Visual diagnostics for JEPA training (which predicts features, not pixels)."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


def expand_token_mask(mask: torch.Tensor, frames: int, image_size: int,
                      tubelet_size: int, patch_size: int) -> torch.Tensor:
    """Expand flat token decisions into a pixel mask shaped ``[T, H, W]``."""
    grid_t = frames // tubelet_size
    grid_h = grid_w = image_size // patch_size
    grid = mask.view(grid_t, grid_h, grid_w)
    return grid.repeat_interleave(tubelet_size, 0).repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)


def save_input_and_mask(video: torch.Tensor, target_mask: torch.Tensor, path: str | Path,
                        tubelet_size: int, patch_size: int) -> None:
    """Save original frames above the sparse-context frames used by the encoder."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    clip = video.detach().cpu()
    pixel_mask = expand_token_mask(target_mask.detach().cpu(), clip.size(1), clip.size(2), tubelet_size, patch_size)
    masked = clip.clone()
    # Muted grey shows missing target regions while preserving visible context.
    masked[:, pixel_mask] = 0.18
    original_frames, masked_frames = clip.permute(1, 0, 2, 3), masked.permute(1, 0, 2, 3)
    save_image(make_grid(torch.cat([original_frames, masked_frames]), nrow=clip.size(1), padding=2), path)


@torch.no_grad()
def save_prediction_similarity(model, video: torch.Tensor, target_mask: torch.Tensor,
                               path: str | Path, tubelet_size: int, patch_size: int) -> float:
    """Visualize predicted-vs-target cosine similarity at every masked tubelet.

    JEPA has no pixel decoder, so this is the faithful qualitative result: bright
    target regions are latent features the predictor matched well; grey regions
    were visible context and were not prediction targets.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    predicted, target = model(video, target_mask)
    similarity = F.cosine_similarity(predicted[0], target[0], dim=-1)
    token_values = torch.full((target_mask.size(1),), 0.35, device=video.device)
    token_values[target_mask[0]] = (similarity + 1) / 2  # map [-1, 1] to displayable [0, 1]
    frames = video.size(2); image_size = video.size(3)
    pixels = expand_token_mask(token_values, frames, image_size, tubelet_size, patch_size)
    maps = pixels.unsqueeze(1).repeat(1, 3, 1, 1).cpu()
    save_image(make_grid(maps, nrow=frames, padding=2), path)
    model.train(was_training)
    return similarity.mean().item()
