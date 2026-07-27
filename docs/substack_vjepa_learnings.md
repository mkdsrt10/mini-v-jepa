# What I Learned Building V-JEPA From Scratch: When Better Prediction Doesn’t Mean Better Representations

I built Mini-V-JEPA to understand video self-supervised learning from the inside out—not just to run a published checkpoint.

The project started small: a Conv3D tubelet embedder, a compact Video ViT encoder, an EMA target encoder, a predictor, and V-JEPA-style spatiotemporal masking. It eventually grew into a training and evaluation pipeline running on the full Something-Something V2 dataset: 168,913 videos across 174 action classes.

The most important thing I learned was not architectural.

> A model can get better at predicting its target features while becoming less specific about which video those features came from.

That sounds subtle, but it changes how you evaluate self-supervised video models.

## The basic idea behind V-JEPA

V-JEPA does not reconstruct pixels.

Instead, it hides regions of a video and asks a predictor to infer the hidden region’s representation from the visible context. The target representation comes from a second encoder updated as an exponential moving average (EMA) of the online encoder.

The simplified pipeline looks like this:

```text
video clip
  ↓
Conv3D tubelet embedding
  ↓
Video Transformer encoder
  ↓
visible context representations
  ↓
predictor
  ↓
predicted hidden-token representations
```

The key distinction is that the model predicts in feature space, not RGB space. In theory, this should encourage it to learn motion, object interactions, and scene structure rather than memorize low-level pixels.

## Building the system incrementally

I deliberately built the project in milestones.

First came synthetic moving shapes: squares, circles, collisions, occlusions, and direction changes. Synthetic data was useful because debugging video learning is much easier when the motion is visible and the ground truth is obvious.

Then I added:

- Conv3D patchification for video inputs shaped `[B, C, T, H, W]`
- a compact Video ViT encoder
- V-JEPA-style full-temporal tube masks
- an EMA target encoder
- a feature-space prediction loss
- checkpointing and training metrics
- linear probing, k-NN classification, and retrieval evaluation
- mask and embedding visualizations

Only after the pipeline could overfit synthetic data did I move to Something-Something V2.

That sequencing mattered. It prevented me from confusing a dataset, decoder, masking, model, or evaluation bug with a “hard learning problem.”

## The first misleading metric: target cosine similarity

Early on, training looked excellent.

Loss fell quickly. Predicted target embeddings had very high cosine similarity with EMA target embeddings. On some runs, target cosine similarity approached `0.99`.

At first glance, that seems like a success.

But it is not enough.

If most target embeddings are similar to one another, a predictor can produce a generic feature vector that scores well against the correct target without learning much about the specific video.

So I added a stronger diagnostic.

## Correct target versus wrong target

For every prediction, I compare two cosine similarities:

```text
correct-target cosine = prediction vs. its own video's target features
wrong-target cosine   = prediction vs. another video's target features
margin                = correct − wrong
```

The wrong target is sampled from another video in the same batch, after target features are computed. Both comparisons use the same masked token positions.

This produces a simple question:

> Does the model predict the right video better than a different video?

That margin was much more informative than raw cosine similarity.

A high correct-target cosine is good. But a high correct-target cosine combined with a similarly high wrong-target cosine means the representation may be generic.

## The surprising result

I trained a compact model on the complete Something-Something V2 training set:

- 168,913 videos
- 174 action classes
- 16 frames per clip
- 112×112 resolution
- 128-dimensional encoder
- 3 Transformer blocks
- batch size 16
- full-temporal two-group tube masking

I evaluated checkpoints across training.

| Step | Correct−Wrong Margin | Linear Probe | k-NN | Retrieval R@1 | Retrieval R@5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 0.445 | 12.5% | 13.5% | 12.0% | 42.5% |
| 20k | **0.498** | 11.5% | 13.5% | 12.0% | 44.0% |
| 30k | 0.443 | 15.5% | 13.0% | 12.5% | 48.0% |
| 40k | 0.339 | 12.0% | 16.0% | 13.5% | **53.0%** |
| 50k | 0.292 | 14.0% | 15.0% | 13.0% | 47.5% |
| 100k | 0.246 | **17.0%** | **18.5%** | **18.5%** | **53.0%** |

The result is counterintuitive.

The correct-vs-wrong prediction margin peaked around 20k steps. But downstream action metrics—linear probing, k-NN, and retrieval—continued improving through 100k steps.

In other words:

> The model became less sharply video-specific according to the JEPA prediction diagnostic, while becoming more useful for action recognition and retrieval.

## Why can this happen?

My interpretation is that the model begins by learning video-specific details that make masked feature prediction easier.

Later, it may learn more invariant features: hands, objects, motion categories, camera movement, and interaction patterns. These are useful for recognizing actions across different videos, but they can also make different videos look more similar in embedding space.

That is not necessarily a failure.

It is a reminder that “good representations” depend on the downstream task.

For a video-specific predictive world model, I may prefer the 20k checkpoint with the strongest correct-versus-wrong margin.

For retrieval or action classification, the 100k checkpoint is better.

There is no universally best checkpoint. There are checkpoints that are best for different purposes.

## Effective rank helped rule out collapse

I also tracked embedding effective rank: a measure of how many meaningful directions the representations occupy.

The effective rank grew from roughly 13 at 10k steps to roughly 32 at 100k steps.

That matters because the falling wrong-target margin could otherwise look like representation collapse. But collapse would usually make embeddings less diverse.

Here, downstream metrics and effective rank both improved. The model was not collapsing; it was becoming more invariant.

## What did not work as expected

A few lessons were practical rather than theoretical.

First, GPU memory usage was not the bottleneck. Video decoding was.

The NVIDIA L4 had plenty of unused memory, but OpenCV decoding of VP9 WebM clips limited utilization. Increasing batch size alone did not guarantee faster training. Data loading, worker behavior, prefetching, and decoder overhead mattered more than raw VRAM capacity.

Second, raw JEPA loss was not enough to guide decisions.

A low feature-prediction loss can mean the model is learning useful structure. It can also mean the target features are too easy to predict generically. Without contrastive diagnostics and downstream probes, it is easy to overestimate progress.

Third, masking design matters enormously.

Independent random masks per frame can turn a video task into an image task. I switched to large, coherent spatiotemporal tube masks so that the model had to use context across time and space.

## Where this mini implementation is relative to state of the art

This is not a reproduction of Meta’s large-scale V-JEPA systems.

My model is deliberately small: 128-dimensional embeddings and 3 Transformer blocks, trained at 112×112 resolution. State-of-the-art V-JEPA-style systems use much larger ViTs, stronger augmentation, larger-scale training, more compute, and richer evaluation protocols.

That is not a weakness of the project. It is the point.

The goal was to make the mechanisms inspectable:

- see exactly what is masked
- verify EMA behavior
- detect indexing errors in wrong-target evaluation
- inspect prediction maps at checkpoints
- compare representation quality across training time
- understand why one metric can improve while another degrades

## What I would improve next

The next experiments are clear.

1. Continue the controlled run from 100k to 200k steps without reheating the learning rate.
2. Add stronger video augmentations: temporal jitter, random resized crops, and color perturbations.
3. Make masking more temporally demanding by mixing full-temporal tube masks with shorter temporal blocks or future-prediction masks.
4. Add attentive pooling and supervised fine-tuning baselines, rather than relying only on mean-pooled frozen features.
5. Scale the encoder from 128 dimensions / 3 blocks toward the original planned 384 dimensions / 6 blocks.

The important part is to change one factor at a time.

## Final takeaway

Building V-JEPA from scratch taught me that self-supervised learning is not just about making a loss decrease.

It is about asking: *what kind of information is the model becoming good at preserving?*

A better prediction score can mean better modeling of the current target. It does not automatically mean better representations for retrieval, classification, or transfer.

The most useful habit I developed was treating every training metric as a hypothesis, not a verdict.

That is how I want to continue this project: not as a black-box training run, but as an experiment in understanding what video representations actually learn.
