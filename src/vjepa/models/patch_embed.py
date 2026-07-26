import torch
from torch import nn


class VideoPatchEmbed(nn.Module):
    """Convert a video into one embedding per small spatiotemporal region.

    A *tubelet* is a ``tubelet_size × patch_size × patch_size`` chunk of a
    video. For example, a 2 × 16 × 16 tubelet contains a 16 × 16 patch across
    two adjacent frames. ``Conv3d`` is used as a learnable, non-overlapping
    cutter: its kernel selects a tubelet and its output channels create the
    embedding for that tubelet.

    Shape example: ``[2, 3, 16, 112, 112] → [2, 392, 384]``.
    """

    def __init__(self, in_channels=3, embed_dim=384, patch_size=16, tubelet_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        # The stride equals the kernel size: tubelets never overlap and every
        # input region is visited exactly once.
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv3d expects video ordered as batch, channels, time, height, width.
        if video.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W] video, received shape {tuple(video.shape)}")
        _, _, frames, height, width = video.shape
        if frames % self.tubelet_size or height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "Video dimensions must be divisible by tubelet_size and patch_size: "
                f"got (T={frames}, H={height}, W={width})"
            )
        # Conv3d output is [B, D, T', H', W'], where D is ``embed_dim`` and
        # each T'×H'×W' location corresponds to one tubelet.
        x = self.proj(video)

        # Transformers expect a sequence: flatten the 3D tubelet grid into N
        # tokens, then move D to the last axis: [B, D, T', H', W'] → [B, N, D].
        return x.flatten(2).transpose(1, 2)
