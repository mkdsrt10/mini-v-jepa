import math


def cosine_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    progress = min(step / max(total_steps, 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


def warmup_cosine_learning_rate(
    step: int,
    total_steps: int,
    peak: float,
    end: float,
    warmup_steps: int,
    start: float | None = None,
) -> float:
    """Linearly warm up, then cosine-decay the learning rate.

    ``step`` is one-based. With a 100-step warm-up, update 100 uses the peak
    rate and the final update uses ``end`` exactly.
    """
    start = end if start is None else start
    if warmup_steps > 0 and step <= warmup_steps:
        return start + (peak - start) * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    return end + (peak - end) * (math.cos(math.pi * progress) + 1) / 2
