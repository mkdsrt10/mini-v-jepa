"""Large spatiotemporal target masks used by V-JEPA-style training."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MaskIndices:
    """The two disjoint token sets used in one masked-prediction step.

    Both tensors have shape ``[B, tokens_per_set]`` and contain flattened
    indices into a token grid ordered as ``[time, height, width]``. Dense,
    equally-sized index tensors are deliberate: the context encoder and
    predictor can process a whole batch without padding.
    """

    context_indices: torch.Tensor
    target_indices: torch.Tensor
    target_mask: torch.Tensor


class MultiBlockMask:
    """Mask 70–90% of video tokens using a small number of large 3D blocks.

    The target blocks span time as well as space. Predicting them therefore
    requires the model to reason about motion and object continuity, instead
    of merely copying pixels from an immediately adjacent patch.
    """

    def __init__(self, ratio: float = 0.8, num_blocks: int = 4, max_blocks: int = 8):
        if not 0 < ratio < 1:
            raise ValueError("ratio must be in (0, 1)")
        if num_blocks < 1 or max_blocks < num_blocks:
            raise ValueError("Require 1 <= num_blocks <= max_blocks")
        self.ratio, self.num_blocks, self.max_blocks = ratio, num_blocks, max_blocks

    def _block_shape(self, grid: tuple[int, int, int]) -> tuple[int, int, int]:
        """Choose a large block whose volume is roughly one target-block share."""
        time, height, width = grid
        volume_per_block = math.ceil(time * height * width * self.ratio / self.num_blocks)
        # Use at least half the temporal axis: this is a *video* mask, not an
        # independent image mask on every frame.
        block_time = min(time, max(1, round(time * 0.5)))
        spatial_area = math.ceil(volume_per_block / block_time)
        block_height = min(height, max(1, round(math.sqrt(spatial_area * height / width))))
        block_width = min(width, max(1, math.ceil(spatial_area / block_height)))
        return block_time, block_height, block_width

    def _sample_target_mask(self, grid: tuple[int, int, int], device) -> torch.Tensor:
        time, height, width = grid
        total_tokens = time * height * width
        requested_targets = round(total_tokens * self.ratio)
        block_time, block_height, block_width = self._block_shape(grid)
        mask = torch.zeros(time, height, width, dtype=torch.bool, device=device)

        # Draw a few large cuboids. If two blocks overlap, draw another one so
        # that we still reach the requested masking budget.
        blocks_drawn = 0
        while int(mask.sum()) < requested_targets and blocks_drawn < self.max_blocks:
            start_t = torch.randint(time - block_time + 1, (), device=device).item()
            start_h = torch.randint(height - block_height + 1, (), device=device).item()
            start_w = torch.randint(width - block_width + 1, (), device=device).item()
            mask[start_t:start_t + block_time, start_h:start_h + block_height, start_w:start_w + block_width] = True
            blocks_drawn += 1

        flat = mask.flatten()
        current_targets = int(flat.sum())
        if current_targets > requested_targets:
            # Blocks are the important structure. We only remove a small random
            # edge subset when their union slightly exceeds the exact budget.
            remove = flat.nonzero(as_tuple=False).flatten()[torch.randperm(current_targets, device=device)[:current_targets - requested_targets]]
            flat[remove] = False
        elif current_targets < requested_targets:
            # A tiny grid can exhaust its block placements. Finish the budget
            # from the remaining tokens; normal video grids take the block path.
            available = (~flat).nonzero(as_tuple=False).flatten()
            add = available[torch.randperm(available.numel(), device=device)[:requested_targets - current_targets]]
            flat[add] = True
        return flat

    def sample_indices(self, batch_size: int, grid: tuple[int, int, int], device=None) -> MaskIndices:
        """Return context and target token indices, plus the equivalent bool mask."""
        target_mask = torch.stack([self._sample_target_mask(grid, device) for _ in range(batch_size)])
        target_indices = target_mask.nonzero(as_tuple=False)[:, 1].view(batch_size, -1)
        context_indices = (~target_mask).nonzero(as_tuple=False)[:, 1].view(batch_size, -1)
        return MaskIndices(context_indices, target_indices, target_mask)

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None) -> torch.Tensor:
        """Compatibility path for the trainer, which consumes a boolean mask."""
        return self.sample_indices(batch_size, grid, device).target_mask


class VJEPATubeMask:
    """The two full-temporal multi-block mask groups from V-JEPA v1.

    V-JEPA's published configuration uses eight 15%-area spatial blocks and
    two 70%-area spatial blocks. Each spatial mask is repeated over the entire
    temporal tubelet axis, so a patch is either visible throughout the clip or
    masked throughout the clip. This prevents the frame-independent masking
    pattern that would weaken temporal prediction.
    """

    SPECS = (
        {"name": "eight_small", "num_blocks": 8, "spatial_scale": 0.15, "aspect_ratio": (0.75, 1.5)},
        # 0.50 is a curriculum-friendly version of V-JEPA's 0.70 large-block
        # scale for our small 7×7 grid. It keeps meaningful context at 112px.
        {"name": "two_large", "num_blocks": 2, "spatial_scale": 0.50, "aspect_ratio": (0.75, 1.5)},
    )

    def _sample_spatial_mask(self, height: int, width: int, spec: dict, device) -> torch.Tensor:
        area = max(1, int(height * width * spec["spatial_scale"]))
        for _ in range(100):
            spatial = torch.zeros(height, width, dtype=torch.bool, device=device)
            for _ in range(spec["num_blocks"]):
                aspect = torch.empty((), device=device).uniform_(*spec["aspect_ratio"]).item()
                block_h = min(height, max(1, round(math.sqrt(area * aspect))))
                block_w = min(width, max(1, round(math.sqrt(area / aspect))))
                top = torch.randint(height - block_h + 1, (), device=device).item()
                left = torch.randint(width - block_w + 1, (), device=device).item()
                spatial[top:top + block_h, left:left + block_w] = True
            if spatial.any() and (~spatial).any():
                return spatial
        raise RuntimeError("Unable to sample a non-empty V-JEPA context mask")

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None) -> list[torch.Tensor]:
        duration, height, width = grid
        masks = []
        for spec in self.SPECS:
            # One geometry per group and batch keeps token counts equal. The
            # geometry changes at the next call, while every frame is a tube.
            spatial = self._sample_spatial_mask(height, width, spec, device)
            tube_mask = spatial.unsqueeze(0).expand(duration, -1, -1).flatten()
            masks.append(tube_mask.unsqueeze(0).expand(batch_size, -1).clone())
        return masks
