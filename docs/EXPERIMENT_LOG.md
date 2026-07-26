# Mini-V-JEPA experiment log

This document records the current implementation, the first real
Something-Something V2 experiments, what the metrics mean, and the decisions
they motivate. It is deliberately a compact research log rather than a claim
of a production-scale V-JEPA reproduction.

## Goal

Mini-V-JEPA is a readable V-JEPA-style learning project. The model receives a
video, hides spatiotemporal tubelets, and predicts **target-encoder latent
features** for the hidden regions. It does not reconstruct RGB pixels.

The educational goals are to make the full workflow inspectable:

1. Video tubelet embedding with a Conv3d patchifier.
2. Video ViT context and EMA target encoders.
3. Large full-temporal V-JEPA-style spatial block masks.
4. A predictor trained with normalized latent MSE.
5. Checkpoints, visual diagnostics, and representation-focused evaluation.

## Data and baseline setup

The first real-data run uses a balanced subset of Something-Something V2:

- 10 action classes
- 100 training clips per class (1,000 clips total)
- 16 frames per clip at 112 x 112 pixels
- batch size 2
- Video ViT: 128 embedding dimensions, depth 3, 4 attention heads
- full-temporal two-group tube masks

The two mask groups follow the idea of V-JEPA's mixed masking scales. On this
smaller 7 x 7 spatial token grid, the large group was softened to roughly 50%
area rather than the original V-JEPA v1 70% scale, so that useful context
remains visible. Actual batch mask ratio varies with sampled block geometry;
the observed range is roughly 45--82%.

## Training instrumentation

Every newly logged training point records:

- JEPA latent prediction loss
- actual target-mask ratio
- correct-target cosine similarity
- EMA momentum
- learning rate actually used for the update
- online-to-EMA encoder RMS parameter distance
- target and predictor feature standard deviations

The RMS parameter distance is

`sqrt(mean((theta_online - theta_EMA)^2))`.

It is measured after the EMA update. A small non-zero value is expected: it
means the target is a delayed teacher. A large increase means online weights
are changing faster than the target can track.

`scripts/plot_training_history.py` turns these records into a six-panel run
dashboard.

## Why absolute cosine was not enough

Early runs showed high target cosine and low JEPA loss, but those values alone
can be misleading. A predictor may match generic, position-dependent target
features without knowing which video it is processing.

The primary discriminating metric is therefore:

`correct-target cosine - wrong-target cosine`

For every held-out video, the prediction is compared both to its real target
features and to target features from a different video at the same token
positions. A larger positive margin means the representation is more specific
to the actual video rather than merely generic or positional.

## Scheduled 2,000-step run

Configuration: `configs/pretrain_something_v2_1k_ema_schedule.yaml`

This fresh run kept data, model, and masks fixed while applying both schedules:

- EMA momentum: cosine ramp from `0.996` to `0.9999`
- learning rate: 100-step linear warm-up from `3e-5` to `3e-4`
- learning rate: cosine decay from `3e-4` to `3e-5` by step 2,000

Selected recorded values:

| Step | EMA momentum | Learning rate | Correct cosine | Encoder EMA RMS distance |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.9960002 | 5.7e-5 | 0.4022 | 0.000288 |
| 100 | 0.9960240 | 3.0e-4 | 0.9670 | 0.000939 |
| 1,000 | 0.9979500 | 1.761e-4 | 0.9221 | 0.001355 |
| 2,000 | 0.9999000 | 3.0e-5 | 0.9744 | 0.001320 |

The full local artifact set is intentionally ignored by Git because it
contains large checkpoints and dataset-derived outputs. The key paths are:

- `outputs/pretrain_something_v2_1k_ema_cosine_2k/metrics.jsonl`
- `outputs/pretrain_something_v2_1k_ema_cosine_2k/training_history.png`
- `outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_last.pt`

## Matched-checkpoint result

On 100 held-out validation videos, using a fixed late central 3D target block
and full spatiotemporal context:

| Step | Run | Correct cosine | Wrong cosine | Correct-minus-wrong margin |
| ---: | --- | ---: | ---: | ---: |
| 100 | fixed baseline | 0.9828 | 0.9804 | 0.0024 |
| 100 | scheduled | 0.9767 | 0.9714 | 0.0053 |
| 1,100 | fixed baseline | 0.9340 | 0.8683 | 0.0658 |
| 1,100 | scheduled | 0.9377 | 0.8274 | **0.1103** |

At the matched 1,100-step checkpoint, the scheduled run has an approximately
68% larger margin (`0.1103` versus `0.0658`). Its correct cosine is slightly
higher, but the main improvement is lower similarity to the wrong video. This
is evidence of more video-specific target prediction.

Reproduce the comparison with:

```bash
python3 scripts/compare_wrong_target_margin.py \
  --checkpoint outputs/pretrain_something_v2_1k_tube/checkpoint_step_000100.pt \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_000100.pt \
  --checkpoint outputs/pretrain_something_v2_1k_tube/checkpoint_step_001100.pt \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_001100.pt \
  --samples 100 --output outputs/wrong_target_margin_comparison.json
```

## Qualitative diagnostics

Absolute target-cosine maps tend to become white or grey very early and hide
small changes. Future checkpoints now save two diverging diagnostics instead:

- `prediction_contrast_step_*.png`: per-tubelet correct-target cosine minus
  wrong-target cosine. Red means the correct target wins; grey is visible
  context and is not a prediction target.
- `prediction_contrast_change_from_0100_*.png`: contrast improvement relative
  to the fixed step-100 preview.

These visualizations complement the held-out metric; they do not replace it.

## Learnings and current limitations

### 1. The objective has an early generic-feature shortcut

At step 100, correct and wrong targets are almost equally similar. High
absolute cosine therefore did not prove useful video understanding. The
wrong-target margin is now the primary health metric.

### 2. The schedules improved specificity

The scheduled run increases the held-out margin at step 1,100. The result is
promising, but it is a **combined** EMA-and-learning-rate experiment. It does
not isolate which schedule contributed the improvement.

### 3. Data diversity is the most likely remaining bottleneck

The 1,000-clip subset is enough to debug the pipeline, but not enough to make
strong claims about action-level representation learning. At 2,000 steps and
batch size 2, training processes about 4,000 clips: roughly four passes over
the subset.

### 4. Model capacity is not the first suspected blocker

The small encoder can reduce the loss and create a non-trivial held-out margin.
It may eventually limit semantic action features, but current evidence points
more strongly to objective shortcuts and limited data diversity.

### 5. Linear-probe results remain inconclusive

The preliminary small protocol (20 training clips and 10 validation clips per
class) produced around 6--7% top-1 accuracy on a 10-way task. That is near
chance and does not yet demonstrate linearly separable action features. Use a
fixed, larger probe split before treating it as a model-selection metric.

## Next steps

1. **Isolate schedules.** Run fixed-EMA plus scheduled-LR and scheduled-EMA
   plus fixed-LR, each from the same seed and for 1,100 steps. Compare margins.
2. **Scale data.** Train on 5,000--10,000 balanced clips while retaining the
   same held-out classes and evaluation protocol.
3. **Keep the margin as the gate.** Report correct cosine, wrong cosine, and
   their difference at every selected checkpoint.
4. **Run a fixed full linear probe.** Use the same train/validation examples,
   classifier epochs, and seeds for all checkpoints.
5. **Increase task difficulty only if needed.** If a larger data run still has
   a small margin or poor probe accuracy, strengthen masks or reduce context
   while checking that target feature standard deviation remains healthy.

## Reproducibility checklist

- Set a fixed seed in the YAML configuration.
- Use a dedicated `output_dir` per experiment; do not mix metrics files.
- Compare same-step checkpoints on the same held-out samples.
- Retain the configuration stored within every checkpoint.
- Record whether the run used CUDA or CPU. Hardware changes wall-clock time
  and can introduce small numerical differences, even with the same seed.
