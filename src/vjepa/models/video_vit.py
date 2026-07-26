import torch
from torch import nn

from .patch_embed import VideoPatchEmbed


class VideoViT(nn.Module):
    """Video transformer encoder for fixed-size clips.

    Input clips use ``[B, C, T, H, W]``. Conv3d patchification produces a
    spatiotemporal token grid, learned positional embeddings preserve each
    token's location, and Transformer blocks contextualize the full video.
    """
    def __init__(self, image_size=224, num_frames=None, patch_size=16, tubelet_size=2,
                 embed_dim=384, depth=8, num_heads=None, *, frames=None, heads=None):
        super().__init__()
        # ``frames`` and ``heads`` remain accepted during the scaffold's migration.
        # New code should use the clearer ``num_frames`` and ``num_heads`` names.
        num_frames = num_frames if num_frames is not None else (frames if frames is not None else 16)
        num_heads = num_heads if num_heads is not None else (heads if heads is not None else 6)
        if num_frames % tubelet_size or image_size % patch_size:
            raise ValueError("num_frames/image_size must divide exactly into tubelets/patches")
        self.image_size, self.num_frames = image_size, num_frames
        self.patch_size, self.tubelet_size = patch_size, tubelet_size
        # Stage 1: turn pixels into a sequence of learnable tubelet embeddings.
        self.patch_embed = VideoPatchEmbed(3, embed_dim, patch_size, tubelet_size)

        # Number of tokens in one video. For the 112px reference model this is
        # (16 / 2) × (112 / 16) × (112 / 16) = 8 × 7 × 7 = 392.
        tokens = (num_frames // tubelet_size) * (image_size // patch_size) ** 2

        # A transformer alone does not know where a token came from. This
        # learned table gives every temporal/spatial tubelet a location signal.
        # The leading dimension of 1 lets PyTorch share it across every batch.
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens, embed_dim))

        # Each block mixes information between all tubelets (self-attention),
        # then transforms each token independently (the MLP inside the block).
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, video: torch.Tensor, visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        # [B, C, T, H, W] → [B, N, D], then add the location information.
        x = self.patch_embed(video) + self.pos_embed
        if visible_mask is not None:
            # V-JEPA supplies this mask: False tokens are prediction targets,
            # so the context encoder only receives the visible tokens. Each
            # video in a batch must have the same number of visible tokens.
            x = x[visible_mask].view(x.shape[0], -1, x.shape[-1])
        # Contextual video representations, still one D-dimensional vector per
        # input token: [B, N_visible, D].
        return self.norm(self.encoder(x))
