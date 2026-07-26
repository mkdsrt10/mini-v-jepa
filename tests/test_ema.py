import torch
from torch import nn
from vjepa.training.ema import parameter_rms_distance, update_ema
from vjepa.training.schedules import cosine_momentum, warmup_cosine_learning_rate
from vjepa.models.vjepa import VJEPA


def test_ema_moves_target_toward_online_weights():
    online, target = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    online.weight.data.fill_(1); target.weight.data.zero_()
    update_ema(online, target, 0.9)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.1))


def test_parameter_rms_distance_measures_encoder_lag():
    online, target = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    online.weight.data.fill_(3); target.weight.data.fill_(1)
    assert parameter_rms_distance(online, target) == 2.0


def test_cosine_ema_schedule_increases_from_start_to_end():
    values = [cosine_momentum(step, 100, start=0.996, end=0.9999) for step in (0, 50, 100)]
    assert values == sorted(values)
    assert values[0] == 0.996
    assert values[-1] == 0.9999


def test_warmup_cosine_learning_rate_has_requested_endpoints():
    schedule = lambda step: warmup_cosine_learning_rate(
        step, total_steps=2000, peak=3e-4, end=3e-5, warmup_steps=100, start=3e-5
    )
    assert schedule(1) > 3e-5
    assert schedule(100) == 3e-4
    assert schedule(2000) == 3e-5


def test_vjepa_target_encoder_remains_in_eval_mode_during_training():
    model = VJEPA(image_size=32, num_frames=4, patch_size=16, tubelet_size=2,
                  embed_dim=32, depth=1, num_heads=4, predictor_depth=1)
    model.train()
    assert model.context_encoder.training
    assert not model.target_encoder.training
