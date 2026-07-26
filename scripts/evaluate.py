#!/usr/bin/env python3
"""Held-out JEPA prediction evaluation for a synthetic checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.collator import collate_videos
from vjepa.data.video_dataset import MovingMNISTDataset, MovingShapesDataset, SomethingSomethingV2Dataset
from vjepa.losses.jepa_loss import jepa_loss
from vjepa.masking.multiblock import MultiBlockMask, VJEPATubeMask
from vjepa.masking.random_tube import RandomTubeMask
from vjepa.models.vjepa import VJEPA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_config, model_config = config["data"], config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VJEPA(**model_config).to(device)
    model.load_state_dict(checkpoint["trainer"]["model"])
    model.eval()
    dataset_name = data_config.get("dataset", "synthetic")
    if dataset_name == "moving_mnist":
        dataset = MovingMNISTDataset(data_config["root"], data_config["clip_frames"])
        dataset_size = min(args.samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(dataset_size))
    elif dataset_name == "something_v2":
        # Evaluate on held-out validation videos from the same selected classes.
        per_class = max(1, (args.samples + data_config["num_classes"] - 1) // data_config["num_classes"])
        dataset = SomethingSomethingV2Dataset(
            data_config["root"], "validation", data_config["clip_frames"], data_config["image_size"],
            data_config["num_classes"], per_class, data_config.get("class_templates"),
        )
    else:
        dataset = MovingShapesDataset(args.samples, data_config["clip_frames"], data_config["image_size"],
                                      data_config["object_size"], data_config["direction_change_prob"],
                                      data_config["occlusion"], data_config["collisions"], seed=config["seed"] + 1_000_000,
                                      num_classes=data_config.get("num_classes", 10))
    loader = DataLoader(dataset, batch_size=data_config["batch_size"], collate_fn=collate_videos)
    mask_config = config["mask"]
    if mask_config["type"] == "vjepa_tube":
        masker = VJEPATubeMask()
    elif mask_config["type"] == "multiblock":
        masker = MultiBlockMask(mask_config["ratio"], mask_config.get("num_blocks", 4))
    else:
        masker = RandomTubeMask(mask_config["ratio"])
    frames = model_config.get("num_frames", model_config.get("frames"))
    grid = (frames // model_config["tubelet_size"], model_config["image_size"] // model_config["patch_size"], model_config["image_size"] // model_config["patch_size"])
    losses, similarities = [], []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            target_masks = masker(video.size(0), grid, device)
            target_masks = target_masks if isinstance(target_masks, list) else [target_masks]
            for target_mask in target_masks:
                predicted, target = model(video, target_mask)
                losses.append(jepa_loss(predicted, target).item())
                similarities.append(torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean().item())
    result = {"samples": args.samples, "jepa_loss": sum(losses) / len(losses), "target_cosine_similarity": sum(similarities) / len(similarities)}
    output = args.checkpoint.parent / "evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(f"Saved {output}")


if __name__ == "__main__":
    main()
