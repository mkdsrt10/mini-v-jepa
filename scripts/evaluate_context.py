#!/usr/bin/env python3
"""Measure how V-JEPA prediction quality changes with available context."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretrain import load_config
from vjepa.data.collator import collate_videos
from vjepa.data.video_dataset import MovingMNISTDataset, MovingShapesDataset, SomethingSomethingV2Dataset
from vjepa.losses.jepa_loss import jepa_loss
from vjepa.models.vjepa import VJEPA


def dependency_masks(batch_size: int, grid: tuple[int, int, int], device):
    """One late 3D target block and three controlled context choices.

    The target occupies the last half of the clip and a central spatial block.
    Full context has all remaining tokens. Spatial-only context uses remaining
    patches at the same times; past-only context uses all patches before it.
    """
    duration, height, width = grid
    start_t = duration // 2
    block_h, block_w = max(1, height // 2), max(1, width // 2)
    top, left = (height - block_h) // 2, (width - block_w) // 2
    target_3d = torch.zeros(duration, height, width, dtype=torch.bool, device=device)
    target_3d[start_t:, top:top + block_h, left:left + block_w] = True
    target = target_3d.flatten().unsqueeze(0).expand(batch_size, -1)
    full = ~target
    spatial_only_3d = torch.zeros_like(target_3d)
    spatial_only_3d[start_t:] = ~target_3d[start_t:]
    past_only_3d = torch.zeros_like(target_3d)
    past_only_3d[:start_t] = True
    return target, {"full_spatiotemporal": full,
                    "spatial_same_window": spatial_only_3d.flatten().unsqueeze(0).expand(batch_size, -1),
                    "past_only": past_only_3d.flatten().unsqueeze(0).expand(batch_size, -1),
                    "no_context": torch.zeros_like(target)}


def make_dataset(config: dict, samples: int):
    data = config["data"]; name = data.get("dataset", "synthetic")
    if name == "something_v2":
        per_class = max(1, (samples + data["num_classes"] - 1) // data["num_classes"])
        return SomethingSomethingV2Dataset(
            data["root"], "validation", data["clip_frames"], data["image_size"],
            data["num_classes"], per_class, data.get("class_templates"),
        )
    if name == "moving_mnist":
        return Subset(MovingMNISTDataset(data["root"], data["clip_frames"]), range(samples))
    return MovingShapesDataset(samples, data["clip_frames"], data["image_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]; model_config, data_config = config["model"], config["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VJEPA(**model_config).to(device)
    model.load_state_dict(checkpoint["trainer"]["model"]); model.eval()
    dataset = make_dataset(config, args.samples)
    loader = DataLoader(dataset, batch_size=data_config["batch_size"], collate_fn=collate_videos)
    frames = model_config.get("num_frames", model_config.get("frames"))
    grid = (frames // model_config["tubelet_size"], model_config["image_size"] // model_config["patch_size"], model_config["image_size"] // model_config["patch_size"])
    scores = {name: {"loss": [], "cosine_similarity": [], "wrong_target_cosine_similarity": []}
              for name in ("full_spatiotemporal", "spatial_same_window", "past_only", "no_context")}
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            target, contexts = dependency_masks(video.size(0), grid, device)
            for name, context in contexts.items():
                predicted, expected = model(video, target, context)
                scores[name]["loss"].append(jepa_loss(predicted, expected).item())
                scores[name]["cosine_similarity"].append(F.cosine_similarity(predicted, expected, dim=-1).mean().item())
                # Negative control: target features belong to a different clip
                # but occupy identical token positions. High similarity here
                # would expose a position-only or collapsed prediction shortcut.
                if video.size(0) > 1:
                    wrong_target = expected.roll(shifts=1, dims=0)
                    scores[name]["wrong_target_cosine_similarity"].append(
                        F.cosine_similarity(predicted, wrong_target, dim=-1).mean().item())
    result = {"samples": len(dataset), "target": "late central 3D block", "conditions": {
        name: {metric: sum(values) / len(values) if values else None for metric, values in measures.items()} for name, measures in scores.items()}}
    output = args.checkpoint.parent / "context_dependency.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(f"Saved {output}")


if __name__ == "__main__":
    main()
