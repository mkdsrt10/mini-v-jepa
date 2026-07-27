import torch
import torch.nn.functional as F

from ..losses.jepa_loss import jepa_loss
from .ema import parameter_rms_distance, update_ema


class Trainer:
    def __init__(self, model, optimizer, masker, grid, ema_momentum=0.996, device="cpu", ema_schedule=None, learning_rate_schedule=None):
        self.model, self.optimizer, self.masker = model.to(device), optimizer, masker
        self.grid, self.ema_momentum, self.device = grid, ema_momentum, device
        # The callable receives a one-based training step.  It lets the target
        # become progressively slower without changing the rest of the run.
        self.ema_schedule = ema_schedule
        self.learning_rate_schedule = learning_rate_schedule
        self.global_step = 0

    def step(self, video: torch.Tensor) -> dict[str, float]:
        """Run one full JEPA optimization step and return useful health metrics."""
        self.model.train()
        video = video.to(self.device)
        # Set the learning rate before forward/backward so the logged value is
        # the rate actually used by this optimizer update.
        if self.learning_rate_schedule is not None:
            learning_rate = self.learning_rate_schedule(self.global_step + 1)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        # The motion-aware ablation policy derives target blocks from raw frame
        # differences. Existing maskers remain video-independent.
        if getattr(self.masker, "requires_video", False):
            masks = self.masker(video.size(0), self.grid, video.device, video=video)
        else:
            masks = self.masker(video.size(0), self.grid, video.device)
        masks = masks if isinstance(masks, list) else [masks]
        losses, target_stds, prediction_stds, mask_ratios, correct_cosines = [], [], [], [], []
        for mask in masks:
            predicted, target = self.model(video, mask)
            losses.append(jepa_loss(predicted, target))
            target_stds.append(target.float().std().item())
            prediction_stds.append(predicted.detach().float().std().item())
            mask_ratios.append(mask.float().mean().item())
            correct_cosines.append(F.cosine_similarity(predicted.detach(), target, dim=-1).mean().item())
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        ema_momentum = (
            self.ema_schedule(self.global_step + 1)
            if self.ema_schedule is not None
            else self.ema_momentum
        )
        update_ema(self.model.context_encoder, self.model.target_encoder, ema_momentum)
        # A rising value means the online encoder is changing faster than the
        # EMA target can follow; exactly zero would mean no EMA lag at all.
        encoder_ema_parameter_distance = parameter_rms_distance(
            self.model.context_encoder, self.model.target_encoder
        )
        self.global_step += 1
        # Target standard deviation is a cheap collapse check: values close to
        # zero would mean all target representations are becoming identical.
        return {
            "loss": loss.item(),
            # "target mask ratio" is the fraction of tubelet tokens predicted,
            # averaged over V-JEPA's two mask groups for this actual batch.
            "mask_ratio": sum(mask_ratios) / len(mask_ratios),
            "target_mask_ratio": sum(mask_ratios) / len(mask_ratios),
            "correct_target_cosine": sum(correct_cosines) / len(correct_cosines),
            "ema_momentum": ema_momentum,
            "encoder_ema_parameter_distance": encoder_ema_parameter_distance,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "target_std": sum(target_stds) / len(target_stds),
            "prediction_std": sum(prediction_stds) / len(prediction_stds),
            # A comma-separated description remains JSONL-friendly while
            # revealing the sampled policies in the mixed-mask run.
            "mask_policies": ",".join(getattr(self.masker, "last_policy_names", [])),
        }

    def state_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
