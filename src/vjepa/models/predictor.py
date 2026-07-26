import torch
from torch import nn


class Predictor(nn.Module):
    """Predict masked target features from encoded context and target positions."""

    def __init__(self, embed_dim=384, depth=4, heads=6):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        # target_pos is positional embeddings for the locations to predict.
        target_tokens = self.mask_token.expand(context.size(0), target_pos.size(1), -1) + target_pos
        return self.norm(self.net(torch.cat([context, target_tokens], dim=1))[:, -target_pos.size(1):])

