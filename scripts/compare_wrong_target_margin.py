#!/usr/bin/env python3
"""Compare held-out correct-minus-wrong target cosine across checkpoints."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_context import dependency_masks, make_dataset
from pretrain import load_config
from vjepa.data.collator import collate_videos
from vjepa.models.vjepa import VJEPA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Repeat for each checkpoint to compare.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = torch.load(args.checkpoint[0], map_location="cpu", weights_only=False)
    config = first["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config, data_config = config["model"], config["data"]
    frames = model_config.get("num_frames", model_config.get("frames"))
    grid = (frames // model_config["tubelet_size"],
            model_config["image_size"] // model_config["patch_size"],
            model_config["image_size"] // model_config["patch_size"])
    dataset = make_dataset(config, args.samples)
    loader = DataLoader(dataset, batch_size=data_config["batch_size"], collate_fn=collate_videos)

    models = []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["model"] != model_config:
            raise ValueError("All checkpoints must use the same encoder architecture.")
        model = VJEPA(**model_config).to(device)
        model.load_state_dict(checkpoint["trainer"]["model"]); model.eval()
        models.append((checkpoint_path, model))

    scores = {str(path): {"correct": [], "wrong": []} for path, _ in models}
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            target, contexts = dependency_masks(video.size(0), grid, device)
            for checkpoint_path, model in models:
                predicted, correct = model(video, target, contexts["full_spatiotemporal"])
                wrong = correct.roll(shifts=1, dims=0)
                scores[str(checkpoint_path)]["correct"].append(
                    F.cosine_similarity(predicted, correct, dim=-1).mean().item()
                )
                scores[str(checkpoint_path)]["wrong"].append(
                    F.cosine_similarity(predicted, wrong, dim=-1).mean().item()
                )

    results = []
    for checkpoint_path, _ in models:
        score = scores[str(checkpoint_path)]
        correct, wrong = sum(score["correct"]) / len(score["correct"]), sum(score["wrong"]) / len(score["wrong"])
        results.append({
            "checkpoint": str(checkpoint_path), "correct_target_cosine": correct,
            "wrong_target_cosine": wrong, "correct_minus_wrong_margin": correct - wrong,
        })
    result = {"samples": len(dataset), "target": "late central 3D block; full spatiotemporal context", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
