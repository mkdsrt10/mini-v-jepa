"""Visual diagnostics for JEPA training (which predicts features, not pixels)."""
from pathlib import Path

import matplotlib.pyplot as plt
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
def prediction_contrast(
    model, video: torch.Tensor, wrong_video: torch.Tensor, target_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-target correct, wrong, and correct-minus-wrong cosines.

    The prediction comes from ``video`` in both comparisons.  Only the target
    is changed: the correct target uses the same clip, while the wrong target
    uses a fixed different clip.  Positive contrast is the useful signal: the
    model matches its own video better than an unrelated video.
    """
    was_training = model.training
    model.eval()
    predicted, target = model(video, target_mask)
    wrong_target = model.target_features(wrong_video, target_mask)
    correct = F.cosine_similarity(predicted[0], target[0], dim=-1)
    wrong = F.cosine_similarity(predicted[0], wrong_target[0], dim=-1)
    model.train(was_training)
    return correct.cpu(), wrong.cpu(), (correct - wrong).cpu()


def save_token_contrast_map(
    values: torch.Tensor,
    target_mask: torch.Tensor,
    path: str | Path,
    frames: int,
    image_size: int,
    tubelet_size: int,
    patch_size: int,
    title: str,
    color_limit: float = 0.05,
) -> None:
    """Save a diverging per-tubelet map; masked values are grey context."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    grid_t, grid_h = frames // tubelet_size, image_size // patch_size
    token_map = torch.full((target_mask.size(1),), float("nan"))
    token_map[target_mask[0].cpu()] = values.cpu()
    token_map = token_map.view(grid_t, grid_h, grid_h).repeat_interleave(tubelet_size, 0)
    columns = min(frames, 8); rows = (frames + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    colormap = plt.colormaps["coolwarm"].copy(); colormap.set_bad("#777777")
    image = None
    for frame_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if frame_index < frames:
            image = axis.imshow(
                token_map[frame_index], cmap=colormap, vmin=-color_limit, vmax=color_limit,
                interpolation="nearest",
            )
            axis.set_title(f"frame {frame_index + 1}", fontsize=8)
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="cosine difference")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)
