"""Mask policies used by the controlled SSv2 masking ablation.

The classes here deliberately keep every sample in one mask group at the same
number of target tokens.  ``VJEPA.target_features`` can then retain targets as
a dense ``[batch, targets, dim]`` tensor without padding, while mask geometry
may still differ between videos.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .multiblock import MultiBlockMask, VJEPATubeMask


class CenterVisibleVJEPATubeMask(VJEPATubeMask):
    """V-JEPA tube masks with a visible central interaction region.

    In a chosen fraction of videos, at least ``min_center_visible`` of a
    central square remains context.  This tests whether always hiding the
    hand/object area is unnecessarily starving the predictor of useful action
    context.  The target count is preserved exactly after enforcing the rule.
    """

    def __init__(self, min_center_visible: float = 0.30,
                 constrained_clip_fraction: float = 0.75,
                 center_size: int = 3):
        if not 0 < min_center_visible < 1:
            raise ValueError("min_center_visible must be in (0, 1)")
        if not 0 < constrained_clip_fraction <= 1:
            raise ValueError("constrained_clip_fraction must be in (0, 1]")
        if center_size < 1:
            raise ValueError("center_size must be positive")
        self.min_center_visible = min_center_visible
        self.constrained_clip_fraction = constrained_clip_fraction
        self.center_size = center_size
        # Exposed for tests and audit scripts; it describes the latest call.
        self.last_center_constrained: list[torch.Tensor] = []

    def _center_region(self, height: int, width: int, device) -> torch.Tensor:
        size = min(self.center_size, height, width)
        top, left = (height - size) // 2, (width - size) // 2
        region = torch.zeros(height, width, dtype=torch.bool, device=device)
        region[top:top + size, left:left + size] = True
        return region

    @staticmethod
    def _random_subset(indices: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return indices[:0]
        return indices[torch.randperm(indices.numel(), device=indices.device)[:count]]

    def _match_count(self, spatial: torch.Tensor, target_count: int,
                     enforce_center_visibility: bool) -> torch.Tensor:
        """Adjust a sampled block union without changing its target budget."""
        flat = spatial.flatten().clone()
        center = self._center_region(*spatial.shape, spatial.device).flatten()
        if enforce_center_visibility:
            minimum_visible = math.ceil(center.sum().item() * self.min_center_visible)
            maximum_center_targets = int(center.sum().item()) - minimum_visible
            center_targets = (flat & center).nonzero(as_tuple=False).flatten()
            excess = max(0, center_targets.numel() - maximum_center_targets)
            if excess:
                remove = self._random_subset(center_targets, excess)
                flat[remove] = False
                # Move those targets outside the centre whenever possible.
                candidates = ((~flat) & (~center)).nonzero(as_tuple=False).flatten()
                flat[self._random_subset(candidates, excess)] = True

        current_count = int(flat.sum())
        if current_count > target_count:
            remove = self._random_subset(flat.nonzero(as_tuple=False).flatten(), current_count - target_count)
            flat[remove] = False
        elif current_count < target_count:
            candidates = (~flat).nonzero(as_tuple=False).flatten()
            if enforce_center_visibility:
                # Preserve the centre guarantee while filling the fixed budget.
                outside = candidates[~center[candidates]]
                needed = target_count - current_count
                chosen = self._random_subset(outside, min(needed, outside.numel()))
                flat[chosen] = True
                needed -= chosen.numel()
                if needed:
                    allowed_centre = ((~flat) & center).nonzero(as_tuple=False).flatten()
                    flat[self._random_subset(allowed_centre, needed)] = True
            else:
                flat[self._random_subset(candidates, target_count - current_count)] = True
        return flat.view_as(spatial)

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None) -> list[torch.Tensor]:
        duration, height, width = grid
        masks, constrained_groups = [], []
        constrained_count = round(batch_size * self.constrained_clip_fraction)
        for spec in self.SPECS:
            # A reference draw defines one fixed target count for this group.
            target_count = int(self._sample_spatial_mask(height, width, spec, device).sum())
            constrained = torch.zeros(batch_size, dtype=torch.bool, device=device)
            constrained[self._random_subset(torch.arange(batch_size, device=device), constrained_count)] = True
            spatial_masks = [
                self._match_count(
                    self._sample_spatial_mask(height, width, spec, device), target_count,
                    bool(constrained[index]),
                )
                for index in range(batch_size)
            ]
            spatial = torch.stack(spatial_masks)
            masks.append(spatial.unsqueeze(1).expand(-1, duration, -1, -1).flatten(1).clone())
            constrained_groups.append(constrained)
        self.last_center_constrained = constrained_groups
        return masks


class ShortTemporalBlockMask(MultiBlockMask):
    """Large spatial blocks that persist for only a few tubelet timesteps."""

    def __init__(self, ratio: float = 0.70, num_blocks: int = 12,
                 temporal_span: tuple[int, int] = (2, 4), max_blocks: int = 32):
        super().__init__(ratio=ratio, num_blocks=num_blocks, max_blocks=max_blocks)
        if temporal_span[0] < 1 or temporal_span[0] > temporal_span[1]:
            raise ValueError("temporal_span must be a valid positive range")
        self.temporal_span = temporal_span

    def _block_shape(self, grid: tuple[int, int, int]) -> tuple[int, int, int]:
        time, height, width = grid
        maximum_time = min(time, self.temporal_span[1])
        minimum_time = min(maximum_time, self.temporal_span[0])
        block_time = int(torch.randint(minimum_time, maximum_time + 1, ()).item())
        volume_per_block = math.ceil(time * height * width * self.ratio / self.num_blocks)
        spatial_area = math.ceil(volume_per_block / block_time)
        block_height = min(height, max(1, round(math.sqrt(spatial_area))))
        block_width = min(width, max(1, math.ceil(spatial_area / block_height)))
        return block_time, block_height, block_width


class MotionAwareMask:
    """Hide high-motion 3D regions estimated solely from input frame changes."""

    requires_video = True

    def __init__(self, ratio: float = 0.70, num_blocks: int = 16, max_blocks: int = 48):
        if not 0 < ratio < 1:
            raise ValueError("ratio must be in (0, 1)")
        self.ratio, self.num_blocks, self.max_blocks = ratio, num_blocks, max_blocks

    @staticmethod
    def _motion_scores(video: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        # Downsample before differencing so the scores align exactly with video
        # tubelets. No labels, optical flow model, or learned target features
        # are used: the policy only sees raw input motion.
        grayscale = video.detach().mean(dim=1, keepdim=True)
        low_res = F.interpolate(grayscale, size=grid, mode="trilinear", align_corners=False).squeeze(1)
        changes = (low_res[:, 1:] - low_res[:, :-1]).abs()
        return F.pad(changes, (0, 0, 0, 0, 1, 0)) + 1e-6

    def _sample_one(self, score: torch.Tensor) -> torch.Tensor:
        time, height, width = score.shape
        target_count = round(time * height * width * self.ratio)
        mask = torch.zeros_like(score, dtype=torch.bool)
        flat_score = score.flatten()
        blocks_drawn = 0
        while int(mask.sum()) < target_count and blocks_drawn < self.max_blocks:
            # Square-root weighting still favors motion, while avoiding one
            # bright moving edge monopolising every target block.
            centre = int(torch.multinomial(flat_score.sqrt() / flat_score.sqrt().sum(), 1).item())
            centre_t, remainder = divmod(centre, height * width)
            centre_h, centre_w = divmod(remainder, width)
            block_t = int(torch.randint(2, min(4, time) + 1, ()).item())
            block_h = int(torch.randint(2, min(4, height) + 1, ()).item())
            block_w = int(torch.randint(2, min(4, width) + 1, ()).item())
            start_t = min(max(0, centre_t - block_t // 2), time - block_t)
            start_h = min(max(0, centre_h - block_h // 2), height - block_h)
            start_w = min(max(0, centre_w - block_w // 2), width - block_w)
            mask[start_t:start_t + block_t, start_h:start_h + block_h, start_w:start_w + block_w] = True
            blocks_drawn += 1

        # Match the target budget exactly, retaining the most dynamic cells
        # when a block union overshoots and adding dynamic cells when it falls short.
        flat = mask.flatten()
        current_count = int(flat.sum())
        if current_count > target_count:
            selected = flat.nonzero(as_tuple=False).flatten()
            remove = selected[flat_score[selected].argsort()[:current_count - target_count]]
            flat[remove] = False
        elif current_count < target_count:
            candidates = (~flat).nonzero(as_tuple=False).flatten()
            add = candidates[flat_score[candidates].argsort(descending=True)[:target_count - current_count]]
            flat[add] = True
        return flat.view_as(mask)

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None,
                 video: torch.Tensor | None = None) -> torch.Tensor:
        if video is None:
            raise ValueError("MotionAwareMask requires the input video")
        if video.size(0) != batch_size:
            raise ValueError("video batch size does not match mask batch size")
        scores = self._motion_scores(video, grid)
        return torch.stack([self._sample_one(scores[index]) for index in range(batch_size)]).flatten(1)


class MixedAblationMask:
    """Two masks per batch sampled from standard, short, and motion-aware policies."""

    requires_video = True

    def __init__(self, ratio: float = 0.66, num_masks: int = 2,
                 standard_probability: float = 0.50,
                 short_temporal_probability: float = 0.30,
                 motion_aware_probability: float = 0.20):
        probabilities = torch.tensor([
            standard_probability, short_temporal_probability, motion_aware_probability,
        ], dtype=torch.float)
        if num_masks < 1 or torch.any(probabilities < 0) or not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
            raise ValueError("mask probabilities must be non-negative and sum to one")
        self.num_masks, self.probabilities = num_masks, probabilities
        self.standard = MultiBlockMask(ratio=ratio, num_blocks=4, max_blocks=12)
        self.short_temporal = ShortTemporalBlockMask(ratio=ratio)
        self.motion_aware = MotionAwareMask(ratio=ratio)
        self.last_policy_names: list[str] = []

    def __call__(self, batch_size: int, grid: tuple[int, int, int], device=None,
                 video: torch.Tensor | None = None) -> list[torch.Tensor]:
        policy_index = torch.multinomial(self.probabilities.to(device), self.num_masks, replacement=True)
        masks, names = [], []
        for index in policy_index.tolist():
            if index == 0:
                masks.append(self.standard(batch_size, grid, device)); names.append("standard_multiblock")
            elif index == 1:
                masks.append(self.short_temporal(batch_size, grid, device)); names.append("short_temporal")
            else:
                masks.append(self.motion_aware(batch_size, grid, device, video)); names.append("motion_aware")
        self.last_policy_names = names
        return masks
