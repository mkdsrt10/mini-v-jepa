#!/usr/bin/env python3
"""Save a contact sheet for visual debugging of Stage A synthetic clips."""
from pathlib import Path
import sys
import torch
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.video_dataset import MovingShapesDataset


def main():
    sample = MovingShapesDataset(length=1, frames=16, image_size=96, seed=7)[0]
    frames = sample["video"].permute(1, 0, 2, 3)
    output = Path("outputs/stage_a_shapes.png")
    output.parent.mkdir(exist_ok=True)
    save_image(make_grid(frames, nrow=8, padding=2), output)
    print(f"Saved {output}; positions={tuple(sample['object_positions'].shape)}, velocities={tuple(sample['object_velocities'].shape)}")


if __name__ == "__main__":
    main()
