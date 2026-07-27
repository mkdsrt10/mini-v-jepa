import torch
from vjepa.masking.random_tube import RandomTubeMask
from vjepa.masking.multiblock import MultiBlockMask
from vjepa.masking.multiblock import VJEPATubeMask
from vjepa.masking.ablation import CenterVisibleVJEPATubeMask, MixedAblationMask, MotionAwareMask, ShortTemporalBlockMask


def test_random_tube_reuses_spatial_mask_over_time():
    mask = RandomTubeMask(0.5)(2, (4, 4, 4))
    assert mask.shape == (2, 64)
    reshaped = mask.view(2, 4, 16)
    assert torch.equal(reshaped[:, 0], reshaped[:, 1])


def test_multiblock_returns_dense_context_and_target_indices():
    sampler = MultiBlockMask(ratio=0.8, num_blocks=4)
    indices = sampler.sample_indices(2, (8, 7, 7))
    assert indices.context_indices.dtype == torch.long
    assert indices.target_indices.dtype == torch.long
    assert indices.target_indices.shape == (2, 314)
    assert indices.context_indices.shape == (2, 78)
    assert torch.all(indices.target_mask.gather(1, indices.target_indices))
    assert not torch.any(indices.target_mask.gather(1, indices.context_indices))


def test_vjepa_tube_masks_are_identical_across_time():
    masks = VJEPATubeMask()(2, (8, 7, 7))
    assert len(masks) == 2
    for mask in masks:
        tube = mask.view(2, 8, 7 * 7)
        assert torch.equal(tube[:, 0], tube[:, -1])
        assert 0 < mask.float().mean() < 1


def test_centre_visible_tubes_preserve_a_minimum_context_region():
    torch.manual_seed(3)
    sampler = CenterVisibleVJEPATubeMask(min_center_visible=0.30, constrained_clip_fraction=0.75, center_size=3)
    masks = sampler(8, (8, 7, 7))
    centre = torch.zeros(7, 7, dtype=torch.bool); centre[2:5, 2:5] = True
    for group, constrained in zip(masks, sampler.last_center_constrained):
        spatial = group.view(8, 8, 7, 7)[:, 0]
        # All videos in a group have equally dense target tensors.
        assert torch.unique(spatial.flatten(1).sum(dim=1)).numel() == 1
        for index in constrained.nonzero(as_tuple=False).flatten():
            visible_fraction = (~spatial[index][centre]).float().mean().item()
            assert visible_fraction >= 0.30


def test_short_and_motion_aware_masks_have_fixed_target_budgets():
    grid, batch_size = (8, 7, 7), 3
    expected_targets = round(8 * 7 * 7 * 0.70)
    short = ShortTemporalBlockMask(ratio=0.70)(batch_size, grid)
    video = torch.rand(batch_size, 3, 16, 112, 112)
    motion = MotionAwareMask(ratio=0.70)(batch_size, grid, video=video)
    assert torch.all(short.sum(dim=1) == expected_targets)
    assert torch.all(motion.sum(dim=1) == expected_targets)


def test_mixed_ablation_returns_two_dense_mask_groups():
    torch.manual_seed(4)
    video = torch.rand(2, 3, 16, 112, 112)
    masks = MixedAblationMask(ratio=0.70, num_masks=2)(2, (8, 7, 7), video=video)
    assert len(masks) == 2
    assert all(mask.shape == (2, 392) for mask in masks)
    assert all(torch.all(mask.sum(dim=1) == round(392 * 0.70)) for mask in masks)
