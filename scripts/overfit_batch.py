#!/usr/bin/env python
"""The first debugging ritual: verify one batch can overfit."""
import argparse
from pathlib import Path
import sys
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.video_dataset import SyntheticVideoDataset
from vjepa.data.collator import collate_videos
from vjepa.masking.random_tube import RandomTubeMask
from vjepa.masking.multiblock import MultiBlockMask
from vjepa.models.vjepa import VJEPA
from vjepa.training.trainer import Trainer


def load_config(path: Path) -> dict:
    """Load a YAML config and recursively merge its optional ``defaults`` file."""
    config = yaml.safe_load(path.read_text())
    default_name = config.pop("defaults", None)
    if default_name is None:
        return config
    base = load_config(path.parent / default_name)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    torch.manual_seed(config["seed"])
    d, m = config["data"], config["model"]
    loader = DataLoader(SyntheticVideoDataset(
        2, d["clip_frames"], d["image_size"], d.get("object_size", 9),
        d.get("direction_change_prob", 0.1), d.get("occlusion", True), d.get("collisions", True), config["seed"]),
        batch_size=d["batch_size"], collate_fn=collate_videos)
    batch = next(iter(loader))["video"]
    model = VJEPA(**m)
    masker_cls = RandomTubeMask if config["mask"]["type"] == "random_tube" else MultiBlockMask
    frames = m.get("num_frames", m.get("frames"))
    grid = (frames // m["tubelet_size"], m["image_size"] // m["patch_size"], m["image_size"] // m["patch_size"])
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"]), masker_cls(config["mask"]["ratio"]), grid, config["optim"]["ema_momentum"])
    for step in range(config["training"]["steps"]):
        metrics = trainer.step(batch)
        print(f"step {step:03d}  loss={metrics['loss']:.5f}")


if __name__ == "__main__": main()
