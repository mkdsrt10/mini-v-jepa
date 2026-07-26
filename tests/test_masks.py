import torch
from vjepa.masking.random_tube import RandomTubeMask
from vjepa.masking.multiblock import MultiBlockMask
from vjepa.masking.multiblock import VJEPATubeMask


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
