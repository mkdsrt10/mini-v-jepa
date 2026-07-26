#!/usr/bin/env python3
"""Plot V-JEPA training health metrics against global step."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import math


PANELS = (
    ("loss", "JEPA loss"),
    ("ema_momentum", "EMA momentum"),
    ("learning_rate", "Learning rate"),
    ("target_mask_ratio", "Actual target-mask ratio"),
    ("correct_target_cosine", "Correct-target cosine"),
    ("encoder_ema_parameter_distance", "Online–EMA encoder RMS distance"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ema-fallback", type=float, default=0.996)
    parser.add_argument("--lr-fallback", type=float, default=0.0003)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    output = args.output or args.metrics.with_name("training_history.png")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.flatten()
    steps = [record["step"] for record in records]
    for axis, (key, title) in zip(axes, PANELS):
        if key == "target_mask_ratio":
            values = [record.get(key, record.get("mask_ratio", math.nan)) for record in records]
        elif key == "ema_momentum":
            values = [record.get(key, args.ema_fallback) for record in records]
        elif key == "learning_rate":
            values = [record.get(key, args.lr_fallback) for record in records]
        else:
            values = [record.get(key, math.nan) for record in records]
        axis.plot(steps, values, marker="o", markersize=3)
        axis.set_title(title); axis.set_xlabel("Step"); axis.grid(alpha=0.3)
    figure.suptitle("Mini-V-JEPA training history", fontsize=14)
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); print(f"Saved {output}")


if __name__ == "__main__":
    main()
