# Mini-V-JEPA

An approachable, from-scratch implementation of the core ideas behind Meta's V-JEPA: learn a video representation by predicting **latent target features** for masked spatiotemporal regions, not pixels.

Mini-V-JEPA is deliberately compact enough to study end-to-end, while retaining the engineering habits that make research code useful: configuration-driven experiments, modular masking strategies, EMA target updates, smoke tests, and simple evaluation hooks.

> **Scope.** This is an educational implementation inspired by V-JEPA, not an official Meta release or a reproduction of every training detail.

## Why this repository

- A readable path from video clips to patch tokens, masks, latent prediction, and loss.
- Two masking styles: random tubes and multi-block masks.
- Online-context and EMA-target encoders, with a predictor trained in feature space.
- Small scripts that make it easy to overfit one batch before launching a run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/overfit_batch.py --config configs/debug.yaml
```

The debug configuration uses synthetic clips, so no dataset download is required. For a real dataset, point `data.root` at a directory containing class folders of video files, or extend `VideoDataset` for your preferred manifest format.

## Project map

```text
src/vjepa/
  data/          video loading, transforms, batching
  masking/       token-mask generators
  models/        video ViT, predictor, and V-JEPA wrapper
  training/      EMA and learning-rate schedules
  evaluation/    linear-probe, kNN, retrieval utilities
```

## Training flow

1. Sample a video clip and split it into spatiotemporal patches.
2. Mask target tokens; encode only visible context tokens with the online encoder.
3. Encode the full clip with an EMA target encoder (no gradients).
4. Predict target-token embeddings from context embeddings and positional information.
5. Minimize normalized latent-space MSE, then update the target encoder by EMA.

## Milestone 2: Video ViT encoder

`VideoViT` implements the encoder backbone: Conv3d tubelet embeddings, a learned positional embedding for every spatiotemporal patch, six Transformer blocks, and final LayerNorm. The reference configuration in `configs/encoder_112.yaml` maps a `[B, 3, 16, 112, 112]` clip to `[B, 392, 384]` video representations (`8 × 7 × 7` tokens).

## Milestone 3: multi-block masking

`MultiBlockMask` draws a small number of large target cuboids across time and space, then returns three aligned views of the same decision: `target_indices`, `context_indices`, and `target_mask`. With the reference token grid (`8 × 7 × 7 = 392`) and an 80% mask ratio, it produces 314 target tokens and 78 context tokens per video. The model must predict the masked latent features from those sparse visible regions, rather than copy local pixels.

For the Something-Something V2 run, `VJEPATubeMask` uses eight small 15%-area spatial blocks and two large 50%-area spatial blocks, with each mask repeated across the full temporal tubelet axis. The 50% large-block scale is intentionally softened from the original V-JEPA v1 70% setting because this educational run uses a much smaller 7×7 spatial token grid. The two groups are trained as separate feature-prediction objectives. Run `python3 scripts/audit_masks.py` to render a visual audit before a long training run.

## Good first experiments

1. Run `python3 scripts/inspect_synthetic.py` and inspect the generated moving-shapes clips.
2. Run `scripts/overfit_batch.py` and make the loss fall.
3. Change the mask ratio in `configs/debug.yaml`.
4. Compare `random_tube` and `multiblock` masking.
5. Replace synthetic data with a small Something-Something V2 subset.
6. Freeze the encoder and evaluate with a linear probe.

## Stage A: synthetic moving shapes

The default debug dataset is intentionally a small physics world: a square and a circle move over a black background, bounce off walls, occasionally reverse direction, exchange velocities after collisions, and can disappear behind a central occluder. It creates unlimited deterministic clips with per-frame positions, visibility flags, and ground-truth velocity vectors.

The first pretraining configuration contains 10 balanced initial-motion classes with 750 clips per class (7,500 clips total), at 112×112 resolution and 16 frames per clip. Labels are intentionally ignored during self-supervised pretraining and reserved for later evaluation.

This makes the first development stage unusually productive: motion is visible, overfitting is fast, and failures can be traced to a particular clip rather than a downloader or annotation pipeline. Run the inspector to save a contact sheet to `outputs/stage_a_shapes.png`:

```bash
python3 scripts/inspect_synthetic.py
```

Example diagnostic task: train a small probe on frozen clip features to predict each object's next velocity; compare the error for visible objects and objects emerging from the occluder.

### Inspecting a pretraining run

JEPA predicts latent features, not reconstructed RGB pixels. Each run therefore saves an honest visual storyboard: `input_original_and_masked.png` shows the original clip above the sparse context encoder input. At checkpoints, `prediction_contrast_step_*.png` maps **correct-target cosine minus wrong-target cosine** for every predicted tubelet (red is better-than-wrong; grey is visible context). `prediction_contrast_change_from_0100_*.png` shows the improvement relative to step 100. These diverging maps make small, meaningful changes visible where absolute cosine maps saturate. Run held-out feature prediction evaluation after pretraining:

```bash
python3 scripts/evaluate.py --checkpoint outputs/pretrain_synthetic/checkpoint_last.pt
```

### Moving MNIST pretraining

The standard Moving MNIST file uses 20 grayscale frames per clip. Mini-V-JEPA takes a deterministic 16-frame view and repeats grayscale data into RGB channels, allowing the same video encoder to be used unchanged.

```bash
python3 scripts/pretrain.py --config configs/pretrain_moving_mnist.yaml
```

### Context-dependency evaluation

This evaluation predicts one fixed late, central 3D target block while changing only what the context encoder may see: full spatiotemporal context, same-window spatial context, or earlier-frame-only context. It exposes whether a low JEPA loss relies on motion history, local appearance, or both.

```bash
python3 scripts/evaluate_context.py --checkpoint outputs/pretrain_something_v2_1k_tube/checkpoint_last.pt --samples 100
```

Compare the more discriminating held-out metric—correct-target cosine minus
wrong-target cosine—across any matched checkpoints:

```bash
python3 scripts/compare_wrong_target_margin.py \
  --checkpoint outputs/pretrain_something_v2_1k_tube/checkpoint_step_001100.pt \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_001100.pt \
  --samples 100 --output outputs/wrong_target_margin_comparison.json
```

### Linear-probe checkpoint curve

Freeze the encoder, train only one linear classification layer, and use the same 10-class train/validation split at every checkpoint. This is the clearest way to track whether pretraining improves action-relevant features rather than only reducing JEPA loss.

```bash
python3 scripts/linear_probe.py --checkpoint outputs/pretrain_something_v2_1k_tube/checkpoint_last.pt
# Later, compare every saved training checkpoint:
python3 scripts/linear_probe.py --checkpoint-dir outputs/pretrain_something_v2_1k_tube
```

### Training-history plot

Each new training report records JEPA loss, EMA momentum, learning rate, actual target-mask ratio, batch correct-target cosine, and the residual online-to-EMA encoder RMS parameter distance. This final value is the EMA target's post-update lag: a large rise means the online encoder is changing faster than the target can track. Plot them against global step:

```bash
python3 scripts/plot_training_history.py --metrics outputs/pretrain_something_v2_1k_tube/metrics.jsonl
```

### EMA-schedule experiment

The controlled 2,000-step schedule experiment keeps the data, model, and
masks fixed while cosine-ramping target momentum from `0.996` to `0.9999`.
It also uses a 100-step learning-rate warm-up from `3e-5` to `3e-4`, followed
by cosine decay back to `3e-5`. It starts from random initialization in a
separate output directory:

```bash
python3 scripts/pretrain.py --config configs/pretrain_something_v2_1k_ema_schedule.yaml
```

## License

MIT — add a `LICENSE` file before publishing if you choose a different license.
