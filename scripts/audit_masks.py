#!/usr/bin/env python3
"""Generate 100 real-video V-JEPA tube-mask visualizations for inspection."""
import argparse
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pretrain import load_config
from vjepa.data.video_dataset import SomethingSomethingV2Dataset
from vjepa.masking.multiblock import VJEPATubeMask
from vjepa.visualisation.training import save_input_and_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_something_v2_1k.yaml")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="outputs/mask_audit_something_v2")
    args = parser.parse_args()
    config = load_config(Path(args.config)); data, model = config["data"], config["model"]
    dataset = SomethingSomethingV2Dataset(data["root"], data.get("split", "train"), data["clip_frames"],
                                          data["image_size"], data["num_classes"], data["samples_per_class"])
    grid = (model["num_frames"] // model["tubelet_size"], model["image_size"] // model["patch_size"], model["image_size"] // model["patch_size"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest, sampler = [], VJEPATubeMask()
    for number, index in enumerate(random.Random(config["seed"]).sample(range(len(dataset)), min(args.count, len(dataset)))):
        sample = dataset[index]
        for group, mask in enumerate(sampler(1, grid)):
            tube = mask.view(grid[0], grid[1], grid[2])
            temporal_consistent = bool(torch.equal(tube[0], tube[-1]))
            filename = f"{number:03d}_{sample['video_id']}_group{group}.png"
            save_input_and_mask(sample["video"], mask[0], output / filename, model["tubelet_size"], model["patch_size"])
            manifest.append({"file": filename, "video_id": sample["video_id"], "label": sample["template"],
                             "group": VJEPATubeMask.SPECS[group]["name"], "mask_ratio": mask.float().mean().item(),
                             "temporal_consistent": temporal_consistent})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} images; all temporal tubes={all(item['temporal_consistent'] for item in manifest)}")


if __name__ == "__main__":
    main()
