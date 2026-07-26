import torch
from vjepa.models.vjepa import VJEPA
from vjepa.models.patch_embed import VideoPatchEmbed
from vjepa.models.video_vit import VideoViT
from vjepa.data.video_dataset import MovingShapesDataset
from vjepa.data.video_dataset import MovingMNISTDataset


def test_patch_embedding_shape() -> None:
    model = VideoPatchEmbed(
        embed_dim=384,
        tubelet_size=2,
        patch_size=16,
    )

    video = torch.randn(2, 3, 16, 112, 112)
    output = model(video)

    assert output.shape == (2, 392, 384)


def test_video_vit_encoder_shape() -> None:
    model = VideoViT(
        image_size=112,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
    )
    output = model(torch.randn(2, 3, 16, 112, 112))
    assert output.shape == (2, 392, 384)


def test_vjepa_predicts_one_feature_per_target_token():
    model = VJEPA(image_size=32, frames=4, patch_size=16, tubelet_size=2, embed_dim=32, depth=1, heads=4, predictor_depth=1)
    video = torch.randn(2, 3, 4, 32, 32)
    mask = torch.zeros(2, 8, dtype=torch.bool); mask[:, :4] = True
    prediction, target = model(video, mask)
    assert prediction.shape == target.shape == (2, 4, 32)


def test_vjepa_accepts_explicit_context_for_dependency_evaluation():
    model = VJEPA(image_size=32, frames=4, patch_size=16, tubelet_size=2, embed_dim=32, depth=1, heads=4, predictor_depth=1)
    video = torch.randn(2, 3, 4, 32, 32)
    target = torch.zeros(2, 8, dtype=torch.bool); target[:, 4:] = True
    past_only = torch.zeros_like(target); past_only[:, :4] = True
    prediction, expected = model(video, target, past_only)
    assert prediction.shape == expected.shape == (2, 4, 32)


def test_moving_shapes_returns_interpretable_motion_targets():
    sample = MovingShapesDataset(length=1, frames=6, image_size=64, seed=10)[0]
    assert sample["video"].shape == (3, 6, 64, 64)
    assert sample["object_positions"].shape == (6, 2, 2)
    assert sample["object_velocities"].shape == (6, 2, 2)
    assert sample["object_visible"].shape == (6, 2)
    assert sample["video"].max() > 0.9


def test_moving_shapes_uses_balanced_motion_class_labels():
    dataset = MovingShapesDataset(length=20, frames=4, image_size=64, num_classes=10)
    assert [dataset[index]["label"] for index in range(10)] == list(range(10))


def test_moving_mnist_adapter_returns_rgb_video():
    dataset = MovingMNISTDataset("data/moving-mnist", frames=16)
    sample = dataset[0]
    assert sample["video"].shape == (3, 16, 64, 64)
    assert sample["video"].dtype == torch.float32
    assert 0 <= sample["video"].min() <= sample["video"].max() <= 1
