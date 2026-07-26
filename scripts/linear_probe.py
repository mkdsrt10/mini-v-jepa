#!/usr/bin/env python3
"""Evaluate one or more V-JEPA checkpoints with the same frozen linear probe."""
import argparse
import glob
import json
from pathlib import Path
import re
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.collator import collate_videos
from vjepa.data.video_dataset import SomethingSomethingV2Dataset
from vjepa.evaluation.linear_probe import extract_mean_features, fit_linear_probe
from vjepa.models.vjepa import VJEPA


def checkpoint_step(path: Path, checkpoint: dict) -> int:
    return checkpoint.get("trainer", {}).get("global_step", int(re.search(r"(\d+)", path.stem).group(1)) if re.search(r"(\d+)", path.stem) else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, help="Repeat to compare named checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, help="Evaluate every checkpoint_step_*.pt plus checkpoint_last.pt")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional explicit JSON output path")
    args = parser.parse_args()
    paths = args.checkpoint or []
    if args.checkpoint_dir:
        paths.extend(Path(path) for path in glob.glob(str(args.checkpoint_dir / "checkpoint_step_*.pt")))
        last = args.checkpoint_dir / "checkpoint_last.pt"
        if last.exists(): paths.append(last)
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("provide --checkpoint or --checkpoint-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]; data = config["data"]
        classes = data["num_classes"]
        train_data = SomethingSomethingV2Dataset(data["root"], "train", data["clip_frames"], data["image_size"], classes, args.train_per_class)
        val_data = SomethingSomethingV2Dataset(data["root"], "validation", data["clip_frames"], data["image_size"], classes, args.val_per_class)
        train_loader = DataLoader(train_data, batch_size=data["batch_size"], collate_fn=collate_videos)
        val_loader = DataLoader(val_data, batch_size=data["batch_size"], collate_fn=collate_videos)
        model = VJEPA(**config["model"]).to(device)
        model.load_state_dict(checkpoint["trainer"]["model"])
        train_features, train_labels = extract_mean_features(model.context_encoder, train_loader, device)
        val_features, val_labels = extract_mean_features(model.context_encoder, val_loader, device)
        torch.manual_seed(config["seed"])
        accuracy, per_class = fit_linear_probe(train_features, train_labels, val_features, val_labels, classes, args.epochs)
        result = {"checkpoint": str(path), "step": checkpoint_step(path, checkpoint), "train_samples": len(train_data), "validation_samples": len(val_data), "top1_accuracy": accuracy, "per_class_accuracy": per_class}
        results.append(result)
        print(json.dumps(result, indent=2))
    output = args.output or paths[0].parent / "linear_probe_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sorted(results, key=lambda item: item["step"]), indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
