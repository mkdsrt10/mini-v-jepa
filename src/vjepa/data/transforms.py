import torch


def normalize(video: torch.Tensor) -> torch.Tensor:
    """Normalize [C, T, H, W] RGB clips with ImageNet statistics."""
    mean = video.new_tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = video.new_tensor([0.229, 0.224, 0.225])[:, None, None, None]
    return (video - mean) / std

