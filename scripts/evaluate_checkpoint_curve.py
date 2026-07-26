#!/usr/bin/env python3
"""Evaluate a fixed set of V-JEPA checkpoints on one held-out 10-class split."""
import argparse
import json
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.collator import collate_videos
from vjepa.data.video_dataset import SomethingSomethingV2Dataset
from vjepa.evaluation.knn import knn_accuracy
from vjepa.evaluation.linear_probe import fit_linear_probe
from vjepa.evaluation.retrieval import nearest_neighbors
from vjepa.models.vjepa import VJEPA


def step_of(path: Path, checkpoint: dict) -> int:
    return checkpoint["trainer"].get("global_step", int(re.search(r"(\d+)", path.stem).group(1)))


def dataset_for(config: dict, split: str, per_class: int):
    data = config["data"]
    return SomethingSomethingV2Dataset(
        data["root"], split, data["clip_frames"], data["image_size"],
        data["num_classes"], per_class, data.get("class_templates"),
    )


def target_and_context(batch_size: int, grid: tuple[int, int, int], device):
    """Fixed late central target used for all held-out margin measurements."""
    duration, height, width = grid
    target_3d = torch.zeros(duration, height, width, dtype=torch.bool, device=device)
    top, left = (height - height // 2) // 2, (width - width // 2) // 2
    target_3d[duration // 2:, top:top + height // 2, left:left + width // 2] = True
    target = target_3d.flatten().unsqueeze(0).expand(batch_size, -1)
    return target, ~target


@torch.no_grad()
def collect_train_features(models, loader, device):
    features = [[] for _ in models]
    labels = []
    for batch in loader:
        video = batch["video"].to(device); labels.append(batch["label"].cpu())
        for index, (_, model) in enumerate(models):
            tokens = model.context_encoder(video)
            features[index].append(F.normalize(tokens.mean(dim=1), dim=-1).cpu())
    return [torch.cat(items) for items in features], torch.cat(labels)


@torch.no_grad()
def collect_validation_features_and_margin(models, loader, grid, device):
    features = [[] for _ in models]
    labels = []
    margins = [{"correct": 0.0, "wrong": 0.0, "tokens": 0} for _ in models]
    for batch in loader:
        video = batch["video"].to(device); labels.append(batch["label"].cpu())
        target, context = target_and_context(video.size(0), grid, device)
        for index, (_, model) in enumerate(models):
            tokens = model.context_encoder(video)
            features[index].append(F.normalize(tokens.mean(dim=1), dim=-1).cpu())
            if video.size(0) > 1:
                predicted, correct_target = model(video, target, context)
                wrong_target = correct_target.roll(shifts=1, dims=0)
                correct = F.cosine_similarity(predicted, correct_target, dim=-1)
                wrong = F.cosine_similarity(predicted, wrong_target, dim=-1)
                margins[index]["correct"] += correct.sum().item()
                margins[index]["wrong"] += wrong.sum().item()
                margins[index]["tokens"] += correct.numel()
    return [torch.cat(items) for items in features], torch.cat(labels), margins


def retrieval_scores(train_features, train_labels, val_features, val_labels):
    neighbors = nearest_neighbors(val_features, train_features, k=min(5, len(train_features)))
    neighbor_labels = train_labels[neighbors]
    return {
        "retrieval_recall_at_1": (neighbor_labels[:, 0] == val_labels).float().mean().item(),
        "retrieval_recall_at_5": (neighbor_labels == val_labels.unsqueeze(1)).any(dim=1).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in args.checkpoint]
    config = checkpoints[0][1]["config"]; model_config, data = config["model"], config["data"]
    for _, checkpoint in checkpoints[1:]:
        if checkpoint["config"]["model"] != model_config or checkpoint["config"]["data"] != data:
            raise ValueError("All checkpoints must use the same model and data configuration.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path, checkpoint in checkpoints:
        model = VJEPA(**model_config).to(device)
        model.load_state_dict(checkpoint["trainer"]["model"]); model.eval()
        models.append((path, model))

    train_data = dataset_for(config, "train", args.train_per_class)
    val_data = dataset_for(config, "validation", args.val_per_class)
    loader_args = {"batch_size": data["batch_size"], "collate_fn": collate_videos}
    train_features, train_labels = collect_train_features(models, DataLoader(train_data, **loader_args), device)
    frames = model_config.get("num_frames", model_config.get("frames"))
    grid = (frames // model_config["tubelet_size"], model_config["image_size"] // model_config["patch_size"], model_config["image_size"] // model_config["patch_size"])
    val_features, val_labels, margins = collect_validation_features_and_margin(
        models, DataLoader(val_data, **loader_args), grid, device
    )

    results = []
    for index, ((path, checkpoint), train_feature, val_feature, margin) in enumerate(zip(checkpoints, train_features, val_features, margins)):
        torch.manual_seed(config["seed"])
        probe_accuracy, _ = fit_linear_probe(train_feature, train_labels, val_feature, val_labels, data["num_classes"], args.probe_epochs)
        correct = margin["correct"] / margin["tokens"]
        wrong = margin["wrong"] / margin["tokens"]
        results.append({
            "checkpoint": str(path), "step": step_of(path, checkpoint),
            "correct_target_cosine": correct, "wrong_target_cosine": wrong,
            "correct_minus_wrong_margin": correct - wrong,
            "frozen_linear_probe_top1": probe_accuracy,
            "knn_classification_top1": knn_accuracy(train_feature, train_labels, val_feature, val_labels, args.knn_k),
            **retrieval_scores(train_feature, train_labels, val_feature, val_labels),
        })
    result = {
        "protocol": {
            "train_per_class": args.train_per_class, "val_per_class": args.val_per_class,
            "wrong_target": "another video in the same batch at identical masked positions",
            "retrieval": "same-class gallery match at rank 1 or within top 5",
        },
        "results": sorted(results, key=lambda item: item["step"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
