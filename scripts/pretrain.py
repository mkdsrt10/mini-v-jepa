#!/usr/bin/env python3
"""Run a resumable first V-JEPA pretraining pass on synthetic moving shapes."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vjepa.data.collator import collate_videos
from vjepa.data.video_dataset import MovingMNISTDataset, MovingShapesDataset, SomethingSomethingV2Dataset
from vjepa.masking.multiblock import MultiBlockMask, VJEPATubeMask
from vjepa.masking.random_tube import RandomTubeMask
from vjepa.models.vjepa import VJEPA
from vjepa.training.schedules import cosine_momentum, warmup_cosine_learning_rate
from vjepa.training.trainer import Trainer
from vjepa.visualisation.training import save_input_and_mask, save_prediction_similarity


def load_config(path: Path) -> dict:
    """Load YAML and recursively merge an optional local ``defaults`` file."""
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


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def save_checkpoint(path: Path, trainer: Trainer, config: dict) -> None:
    """Save all state required to resume training, including optimizer and EMA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainer": trainer.state_dict(), "config": config}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_synthetic.yaml")
    parser.add_argument("--resume", type=Path, help="Checkpoint created by this script")
    parser.add_argument("--max-steps", type=int, help="Temporary run limit; does not modify the YAML")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    torch.manual_seed(config["seed"])
    data_config, model_config = config["data"], config["model"]
    run_config = config["training"]
    device = choose_device(run_config.get("device", "auto"))

    dataset_name = data_config.get("dataset", "synthetic")
    if dataset_name == "moving_mnist":
        dataset = MovingMNISTDataset(data_config["root"], data_config["clip_frames"])
    elif dataset_name == "something_v2":
        dataset = SomethingSomethingV2Dataset(
            data_config["root"], data_config.get("split", "train"), data_config["clip_frames"],
            data_config["image_size"], data_config.get("num_classes"), data_config.get("samples_per_class"),
        )
        print(f"Selected {len(dataset)} videos across {len(dataset.class_templates)} classes: {dataset.class_counts}")
    else:
        dataset = MovingShapesDataset(
            length=data_config["dataset_size"], frames=data_config["clip_frames"],
            image_size=data_config["image_size"], object_size=data_config["object_size"],
            direction_change_prob=data_config["direction_change_prob"],
            occlusion=data_config["occlusion"], collisions=data_config["collisions"], seed=config["seed"],
            num_classes=data_config.get("num_classes", 10), clips_per_class=data_config.get("clips_per_class"),
        )
    loader = DataLoader(dataset, batch_size=data_config["batch_size"], shuffle=True,
                        collate_fn=collate_videos, drop_last=True)
    model = VJEPA(**model_config)
    masker_type = config["mask"]["type"]
    if masker_type == "vjepa_tube":
        masker = VJEPATubeMask()
    elif masker_type == "multiblock":
        masker = MultiBlockMask(config["mask"]["ratio"], config["mask"].get("num_blocks", 4))
    else:
        masker = RandomTubeMask(config["mask"]["ratio"])
    frames = model_config.get("num_frames", model_config.get("frames"))
    grid = (frames // model_config["tubelet_size"], model_config["image_size"] // model_config["patch_size"], model_config["image_size"] // model_config["patch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"], weight_decay=config["optim"]["weight_decay"])
    ema_schedule = None
    ema_schedule_name = config["optim"].get("ema_schedule", "constant")
    if ema_schedule_name == "cosine":
        ema_start = config["optim"]["ema_momentum"]
        ema_end = config["optim"].get("ema_momentum_end", 1.0)
        ema_total_steps = run_config["steps"]
        ema_schedule = lambda step: cosine_momentum(step, ema_total_steps, ema_start, ema_end)
    elif ema_schedule_name != "constant":
        raise ValueError(f"Unknown EMA schedule: {ema_schedule_name}")
    learning_rate_schedule = None
    lr_schedule_name = config["optim"].get("lr_schedule", "constant")
    if lr_schedule_name == "warmup_cosine":
        learning_rate_schedule = lambda step: warmup_cosine_learning_rate(
            step=step,
            total_steps=run_config["steps"],
            peak=config["optim"]["lr"],
            end=config["optim"].get("lr_end", 0.0),
            warmup_steps=config["optim"].get("warmup_steps", 0),
            start=config["optim"].get("lr_start", config["optim"]["lr"]),
        )
    elif lr_schedule_name != "constant":
        raise ValueError(f"Unknown learning-rate schedule: {lr_schedule_name}")
    trainer = Trainer(
        model, optimizer, masker, grid, config["optim"]["ema_momentum"], device,
        ema_schedule=ema_schedule, learning_rate_schedule=learning_rate_schedule,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.load_state_dict(checkpoint["trainer"])
        print(f"Resumed from {args.resume} at step {trainer.global_step}")

    output_dir = Path(run_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.jsonl"
    # Keep one held-aside clip and one mask constant. Checkpoint visualizations
    # are then directly comparable: only the learned model changes over time.
    preview_video = dataset[0]["video"].unsqueeze(0).to(device)
    preview_mask = masker(1, grid, device)
    preview_mask = preview_mask[0] if isinstance(preview_mask, list) else preview_mask
    save_input_and_mask(preview_video[0], preview_mask[0], output_dir / "input_original_and_masked.png",
                        model_config["tubelet_size"], model_config["patch_size"])
    batches = iter(loader)
    mask_description = "V-JEPA full-temporal two-group tube masks" if masker_type == "vjepa_tube" else f"target mask ratio={config['mask']['ratio']:.0%}"
    print(
        f"Pretraining on {device}; {mask_description}; "
        f"EMA schedule={ema_schedule_name}; LR schedule={lr_schedule_name}"
    )
    total_steps = min(run_config["steps"], args.max_steps) if args.max_steps else run_config["steps"]
    while trainer.global_step < total_steps:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        metrics = trainer.step(batch["video"])
        metrics["step"] = trainer.global_step
        # Always persist the final step too, even for a short smoke run.
        if trainer.global_step % run_config["log_every"] == 0 or trainer.global_step == total_steps:
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print("step {step:04.0f} | loss={loss:.5f} | mask={mask_ratio:.0%} | "
                  "target_std={target_std:.4f} | pred_std={prediction_std:.4f}".format(**metrics))
        if trainer.global_step % run_config["checkpoint_every"] == 0:
            checkpoint_path = output_dir / f"checkpoint_step_{trainer.global_step:06d}.pt"
            save_checkpoint(checkpoint_path, trainer, config)
            similarity = save_prediction_similarity(
                trainer.model, preview_video, preview_mask,
                output_dir / f"prediction_similarity_step_{trainer.global_step:06d}.png",
                model_config["tubelet_size"], model_config["patch_size"],
            )
            print(f"  checkpoint saved; preview target cosine similarity={similarity:.4f}")

    save_checkpoint(output_dir / "checkpoint_last.pt", trainer, config)
    similarity = save_prediction_similarity(
        trainer.model, preview_video, preview_mask, output_dir / "prediction_similarity_final.png",
        model_config["tubelet_size"], model_config["patch_size"],
    )
    print(f"Done. Metrics: {metrics_file}; checkpoint: {output_dir / 'checkpoint_last.pt'}")
    print(f"Final preview target cosine similarity={similarity:.4f}")


if __name__ == "__main__":
    main()
