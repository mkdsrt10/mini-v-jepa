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

### 5. Full frozen linear-probe curve is above chance but mostly flat

A fixed full probe trains only a linear classifier on frozen encoder features,
using 100 training and 20 held-out validation videos per class (1,000 / 200
videos total). Chance accuracy is 10%.

| Checkpoint | Top-1 accuracy |
| --- | ---: |
| scheduled step 100 | 13.5% |
| scheduled step 500 | 13.5% |
| scheduled step 1,000 | **15.0%** |
| scheduled step 1,500 | 13.0% |
| scheduled step 2,000 | 14.5% |
| fixed-baseline step 1,100 | 14.0% |

The scheduled representation is modestly above chance and peaks at step 1,000,
slightly ahead of the previous fixed-schedule reference. The curve is not yet
strong or monotonic; several classes remain difficult for a linear head. This
means the wrong-target margin improved more clearly than action-label
separability. The next data-scale and schedule-ablation experiments should use
this exact frozen-probe protocol.

The implementation decodes each video once while extracting features for all
requested checkpoints, making a fair checkpoint curve practical even when VP9
decoding is the bottleneck:

```bash
python3 scripts/linear_probe.py \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_000100.pt \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_001000.pt \
  --checkpoint outputs/pretrain_something_v2_1k_ema_cosine_2k/checkpoint_step_002000.pt \
  --output outputs/linear_probe_checkpoint_curve_2k.json
```

## CUDA 10k-video checkpoint curve

The 10k-video CUDA run was evaluated on six selected checkpoints with one
fixed protocol: 100 train and 20 held-out validation videos per class. The
margin uses a fixed late central target block and a wrong target from another
video in the same batch at identical masked positions. Retrieval uses the
training split as gallery; Recall@1 and Recall@5 mean that a retrieved gallery
clip has the query's action class.

| Step | Correct-wrong margin | Frozen probe | k-NN (k=20) | Retrieval R@1 | Retrieval R@5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 0.1547 | 13.5% | 12.0% | 13.5% | 41.0% |
| 5,000 | 0.1966 | 13.5% | 10.0% | 9.0% | 43.5% |
| 6,500 | 0.1949 | 13.0% | 12.0% | 13.0% | 43.5% |
| 7,000 | 0.2213 | **15.0%** | 10.5% | **15.0%** | **44.5%** |
| 9,000 | **0.2234** | 14.0% | 11.5% | 11.5% | 44.0% |
| 10,000 | 0.2213 | 14.0% | 10.0% | 12.0% | 43.5% |

Interpretation: video-specific latent prediction improves clearly, then
plateaus around 7k steps. Step 9k has the numerically highest held-out margin,
but step 7k is the more useful current checkpoint for action downstream tasks:
it has the best frozen-probe and retrieval R@1 results. k-NN and retrieval
remain only modestly above the 10% 10-way chance level; R@5 is also only
slightly above the approximately 41% independent-label baseline. The project
therefore has evidence of better video specificity, but not yet strong
action-semantic separation.

Reproduce this curve with:

```bash
python3 scripts/evaluate_checkpoint_curve.py \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_004000.pt \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_005000.pt \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_006500.pt \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_007000.pt \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_009000.pt \
  --checkpoint outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_step_010000.pt \
  --output outputs/pretrain_something_v2_10k_cuda_10k/checkpoint_evaluation_curve.json
```

## Full SSv2 100k-update curve

The compact Mini-V-JEPA was then trained on all 168,913 available SSv2 training
videos (174 classes), with batch size 16 for 100,000 updates. Checkpoints were
evaluated on the same fixed 10-class split as the 10k-video run. Effective rank
is `exp(H(p))` of the centered held-out embedding covariance spectrum; it is a
scale-sensitive diversity diagnostic, not a classification metric.

| Step | Margin | Linear probe | k-NN | R@1 | R@5 | Effective rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.4454 | 12.5% | 13.5% | 12.0% | 42.5% | 13.1 |
| 20,000 | **0.4979** | 11.5% | 13.5% | 12.0% | 44.0% | 23.4 |
| 30,000 | 0.4425 | 15.5% | 13.0% | 12.5% | 48.0% | 29.1 |
| 40,000 | 0.3392 | 12.0% | 16.0% | 13.5% | **53.0%** | 30.5 |
| 50,000 | 0.2916 | 14.0% | 15.0% | 13.0% | 47.5% | 30.4 |
| 100,000 | 0.2464 | **17.0%** | **18.5%** | **18.5%** | **53.0%** | **31.7** |

The held-out wrong-target margin peaks at 20k then declines as wrong-target
cosine rises. This is not representational collapse: effective rank increases
throughout, and action metrics improve to their best values at 100k. Use the
20k checkpoint for the strongest video-specific latent-prediction demo, and
the 100k checkpoint for action classification and retrieval. The mismatch is a
useful research finding: the current JEPA objective's best discriminator of
individual videos is not necessarily its best action-semantic representation.

## Controlled 100k-to-200k continuation

The 100k checkpoint was resumed for a further 100k updates using the same full
SSv2 data, model, masks, batch size, optimizer state, and EMA target. The
original LR and EMA schedules had already reached their terminal values at
100k, so the continuation held LR at `3e-5` and EMA momentum at `0.9999`
instead of accidentally reheating the cosine schedule.

Evaluation used the identical fixed 10-class protocol above: 100 labelled
training videos and 20 held-out validation videos per class. This means the
numbers can be compared directly with the 100k checkpoint.

| Step | Margin | Linear probe | k-NN | R@1 | R@5 | Effective rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.2464 | 17.0% | 18.5% | **18.5%** | 53.0% | 31.7 |
| 125,000 | 0.2418 | 16.5% | 19.5% | **19.5%** | 53.5% | 31.7 |
| 150,000 | 0.2459 | 17.0% | 17.5% | 14.0% | 53.5% | 32.0 |
| 170,000 | 0.2483 | 17.5% | 18.5% | 14.5% | 54.0% | 32.0 |
| 180,000 | 0.2503 | 17.0% | 19.0% | 15.5% | 54.0% | 32.3 |
| 195,000 | **0.2562** | 16.5% | **21.0%** | 15.0% | **54.5%** | **32.4** |
| 200,000 | 0.2550 | **18.5%** | 20.5% | 12.0% | 54.0% | 32.0 |

This is **representation saturation, not collapse**. From 150k onward,
effective rank stays close to 32, R@5 stays near 54%, and action metrics make
only small, non-monotonic changes. At the same time, target/prediction feature
standard deviations remain non-zero and the held-out correct-minus-wrong
margin remains positive. The representation is stable and useful, but this
small model/objective is no longer discovering materially new useful feature
directions.

Checkpoint choice should follow the use case:

- **200k:** best frozen linear-probe result (18.5%); use for an action-classification baseline.
- **195k:** best margin, k-NN, effective rank, and R@5; use for a general embedding/retrieval demo.
- **180k:** marginally best R@1 among late checkpoints, but this difference is only one query on a 200-video validation set and should not be overinterpreted.

The late checkpoints are close enough that future comparisons should use a
larger validation set or repeated class-balanced splits before making a strong
claim about a sub-1% difference.

## Next steps

1. **Run the controlled masking ablation.** `configs/pretrain_something_v2_mask_ablation_{a,b,c}.yaml`
   now keeps the compact 1.24M-trainable-parameter model and all optimization
   settings fixed while comparing full-temporal, centre-visible, and mixed
   motion-aware masking for 50k updates. Evaluate at 10k, 25k, and 50k with
   the fixed 10-class protocol. A 2–3 point probe gain without a rank decrease
   is evidence that masking, rather than capacity, was the bottleneck.
2. **Add video augmentation.** The current dataset path only resizes clips.
   Add temporal jitter, random resized crop, and color perturbation during
   pretraining. This is the most direct way to force invariance across
   appearance changes without changing architecture.
3. **Make temporal prediction harder.** Keep the validated tube-mask path but
   mix in shorter 3D blocks and future-only target blocks. Full-temporal tubes
   can be solved from same-time spatial context; this change tests whether the
   encoder actually needs motion history.
4. **Scale one controlled architecture variable.** After the above baseline,
   increase the encoder from 128 dimensions/3 blocks to the planned 384
   dimensions/6 blocks, keeping resolution, clip length, data, and evaluation
   protocol fixed. Compare at matched clip exposures rather than matched
   update counts.
5. **Keep the margin as a diagnostic, not a gate.** Always report correct
   cosine, wrong cosine, and their difference, but select checkpoints using
   held-out probes and retrieval as well.
6. **Use more reliable evaluation uncertainty.** Increase the held-out set or
   repeat class-balanced splits so small differences in R@1 and probe accuracy
   are not mistaken for meaningful gains.

## Reproducibility checklist

- Set a fixed seed in the YAML configuration.
- Use a dedicated `output_dir` per experiment; do not mix metrics files.
- Compare same-step checkpoints on the same held-out samples.
- Retain the configuration stored within every checkpoint.
- Record whether the run used CUDA or CPU. Hardware changes wall-clock time
  and can introduce small numerical differences, even with the same seed.
