import copy
import torch
from torch import nn

from .video_vit import VideoViT
from .predictor import Predictor


class VJEPA(nn.Module):
    """The V-JEPA wrapper: context encoder, frozen EMA target, and predictor."""
    def __init__(self, **encoder_kwargs):
        super().__init__()
        predictor_depth = encoder_kwargs.pop("predictor_depth", 4)
        # The online/context encoder receives only visible video tubelets and
        # is optimized by gradient descent.
        self.context_encoder = VideoViT(**encoder_kwargs)
        # The target encoder starts as an exact copy. Training updates it slowly
        # with EMA, producing stable latent targets without backpropagation.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = Predictor(encoder_kwargs.get("embed_dim", 384), predictor_depth,
                                   encoder_kwargs.get("num_heads", encoder_kwargs.get("heads", 6)))
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Target features must be stable. In particular, do not let the target
        # Transformer's dropout introduce random noise into the regression goal.
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        """Train online modules while always keeping the EMA target deterministic."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def target_features(self, video: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        # Encode every tubelet, then retain only the masked locations the model
        # is asked to predict. ``no_grad`` makes this a fixed training target.
        return self.target_encoder(video)[target_mask].view(video.size(0), -1, self.context_encoder.pos_embed.size(-1))

    def forward(self, video: torch.Tensor, target_mask: torch.Tensor,
                context_mask: torch.Tensor | None = None):
        """Predict target features from either default or explicitly chosen context.

        ``context_mask`` enables controlled evaluations: it can restrict the
        encoder to only earlier frames or only same-time spatial evidence while
        keeping the target locations fixed.
        """
        # Normal pretraining sees the complement of the target mask. Context
        # dependency evaluation passes a stricter visible-token selection.
        visible = ~target_mask if context_mask is None else context_mask
        context = self.context_encoder(video, visible)
        # Give the predictor position embeddings for the missing locations, so
        # it knows *where* in the video each latent feature should be predicted.
        target_pos = self.context_encoder.pos_embed.expand(video.size(0), -1, -1)[target_mask]
        target_pos = target_pos.view(video.size(0), -1, context.size(-1))
        return self.predictor(context, target_pos), self.target_features(video, target_mask)
