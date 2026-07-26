from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def configure_video_decode_worker(_worker_id: int) -> None:
    """Keep each DataLoader worker's OpenCV decoder to one CPU thread.

    Eight workers on an eight-core host already provide parallel video decode.
    Letting every worker create its own OpenCV thread pool can oversubscribe the
    CPU and starve both decoding and GPU transfer.
    """
    import cv2
    cv2.setNumThreads(1)


@dataclass(frozen=True)
class ObjectState:
    """Pixel-space object state; returned as tensors for supervised diagnostics."""
    position: torch.Tensor
    velocity: torch.Tensor


class MovingShapesDataset(Dataset):
    """Deterministic clips with controllable, interpretable temporal dynamics.

    Each sample contains one square and one circle. Objects bounce from walls,
    may reverse direction, exchange velocity on collision, and pass behind a
    central occluder. ``object_velocities`` is the ground-truth per-frame
    velocity, making this a useful diagnostic dataset as well as a JEPA source.
    """
    def __init__(self, length=10_000, frames=16, image_size=64, object_size=9,
                 direction_change_prob=0.10, occlusion=True, collisions=True, seed=0,
                 num_classes=10, clips_per_class: int | None = None):
        if image_size < object_size * 4:
            raise ValueError("image_size must be at least 4x object_size")
        if clips_per_class is not None:
            length = num_classes * clips_per_class
        self.length, self.frames, self.image_size, self.object_size = length, frames, image_size, object_size
        self.direction_change_prob, self.occlusion, self.collisions, self.seed = direction_change_prob, occlusion, collisions, seed
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.seed + index)
        size, canvas = self.object_size, self.image_size
        centers = torch.randint(size, canvas - size, (2, 2), generator=g).float()
        # Ten balanced, interpretable initial-motion classes. Bounces, direction
        # changes, occlusions, and collisions make each clip non-trivial later.
        motion_class = index % self.num_classes
        patterns = torch.tensor([[2, 0], [0, 2], [-2, 0], [0, -2], [2, 2],
                                 [2, -2], [-2, 2], [-2, -2], [2, 1], [1, 2]], dtype=torch.float)
        first_velocity = patterns[motion_class % len(patterns)]
        velocities = torch.stack([first_velocity, -first_velocity.flip(0)])
        frames = torch.zeros(self.frames, 3, canvas, canvas)
        positions, velocity_trace, visibility = [], [], []
        yy, xx = torch.meshgrid(torch.arange(canvas), torch.arange(canvas), indexing="ij")
        occluder = (slice(canvas // 3, 2 * canvas // 3), slice(3 * canvas // 7, 4 * canvas // 7))

        for t in range(self.frames):
            if t > 0 and torch.rand((), generator=g).item() < self.direction_change_prob:
                object_id = torch.randint(0, 2, (), generator=g).item()
                axis = torch.randint(0, 2, (), generator=g).item()
                velocities[object_id, axis].mul_(-1)
            centers += velocities
            for i in range(2):
                for axis in range(2):
                    if centers[i, axis] < size or centers[i, axis] >= canvas - size:
                        velocities[i, axis].mul_(-1)
                        centers[i, axis].clamp_(size, canvas - size - 1)
            if self.collisions and torch.linalg.vector_norm(centers[0] - centers[1]) < 2 * size:
                velocities[[0, 1]] = velocities[[1, 0]]  # equal-mass elastic proxy

            positions.append(centers.clone()); velocity_trace.append(velocities.clone())
            square = (xx - centers[0, 0]).abs().le(size // 2) & (yy - centers[0, 1]).abs().le(size // 2)
            radius = size // 2
            circle = (xx - centers[1, 0]).square() + (yy - centers[1, 1]).square() <= radius * radius
            frames[t, 0][square] = 0.95; frames[t, 1][square] = 0.25
            frames[t, 1][circle] = 0.85; frames[t, 2][circle] = 0.95
            is_visible = torch.ones(2, dtype=torch.bool)
            if self.occlusion:
                for i, center in enumerate(centers):
                    in_occ = occluder[1].start <= center[0] < occluder[1].stop and occluder[0].start <= center[1] < occluder[0].stop
                    is_visible[i] = not in_occ
                frames[t, :, occluder[0], occluder[1]] = 0.12
            visibility.append(is_visible)

        return {
            "video": frames.permute(1, 0, 2, 3),
            "label": motion_class,
            "object_positions": torch.stack(positions),  # [T, objects, xy]
            "object_velocities": torch.stack(velocity_trace),
            "object_visible": torch.stack(visibility),
        }


class SyntheticVideoDataset(MovingShapesDataset):
    """Backward-compatible alias for the Stage A moving-shapes dataset."""
    pass


class MovingMNISTDataset(Dataset):
    """Moving MNIST clips stored in the standard ``[T, N, H, W]`` NumPy file.

    The source is grayscale, so frames are repeated into three channels to
    match the RGB video encoder without changing its architecture.
    """
    def __init__(self, root: str | Path, frames: int = 16):
        path = Path(root) / "mnist_test_seq.npy"
        if not path.exists():
            raise FileNotFoundError(f"Moving MNIST file not found: {path}")
        self.data = np.load(path, mmap_mode="r")
        if self.data.ndim != 4 or frames > self.data.shape[0]:
            raise ValueError(f"Expected [T, N, H, W] with at least {frames} frames, got {self.data.shape}")
        self.frames = frames

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, index: int) -> dict:
        start = (index * 7) % (self.data.shape[0] - self.frames + 1)
        clip = torch.from_numpy(np.array(self.data[start:start + self.frames, index], copy=True)).float() / 255.0
        return {"video": clip.unsqueeze(0).repeat(3, 1, 1, 1), "label": 0}


class SomethingSomethingV2Dataset(Dataset):
    """Balanced, decoded subset of Something-Something V2 ``.webm`` clips.

    ``num_classes=10`` and ``samples_per_class=100`` create the initial 1,000
    video smoke-training set. The class labels are retained for later probing;
    JEPA pretraining itself does not use them.
    """
    def __init__(self, root: str | Path, split="train", frames=16, image_size=112,
                 num_classes: int | None = None, samples_per_class: int | None = None,
                 class_templates: list[str] | None = None):
        import cv2

        self.cv2 = cv2
        root = Path(root)
        labels_dir, video_dir = root / "labels", root / "20bn-something-something-v2"
        metadata = __import__("json").load((labels_dir / f"{split}.json").open())
        label_ids = __import__("json").load((labels_dir / "labels.json").open())
        templates = [name for name, _ in sorted(label_ids.items(), key=lambda item: int(item[1]))]
        if class_templates is not None:
            unknown = set(class_templates) - set(templates)
            if unknown:
                raise ValueError(f"Unknown Something-Something templates: {sorted(unknown)}")
            if len(class_templates) != len(set(class_templates)):
                raise ValueError("class_templates must not contain duplicates")
            if num_classes is not None and len(class_templates) != num_classes:
                raise ValueError("num_classes must match len(class_templates)")
            self.class_templates = class_templates
        else:
            self.class_templates = templates[:num_classes] if num_classes else templates
        # ``labels.json`` says "Closing something", while annotation metadata
        # says "Closing [something]". Normalize that presentation difference.
        canonical = lambda template: template.replace("[", "").replace("]", "")
        selected = {canonical(template): index for index, template in enumerate(self.class_templates)}
        counts = [0] * len(selected)
        self.samples = []
        for item in metadata:
            label = selected.get(canonical(item["template"]))
            path = video_dir / f"{item['id']}.webm"
            if label is None or not path.exists() or (samples_per_class is not None and counts[label] >= samples_per_class):
                continue
            self.samples.append((path, label, item["template"]))
            counts[label] += 1
            if samples_per_class is not None and all(count >= samples_per_class for count in counts):
                break
        if not self.samples:
            raise RuntimeError(f"No usable {split} videos found under {video_dir}")
        self.frames, self.image_size = frames, image_size
        self.class_counts = counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, label, template = self.samples[index]
        capture = self.cv2.VideoCapture(str(path))
        # VP9 random seeking can fail in OpenCV even for valid files. Decode the
        # short clip sequentially, then pick evenly spaced frames in memory.
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(torch.from_numpy(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)))
        capture.release()
        if not decoded:
            raise RuntimeError(f"Unable to decode any frames from {path}")
        frame_indices = np.linspace(0, len(decoded) - 1, self.frames).round().astype(int)
        frames = [decoded[frame_index] for frame_index in frame_indices]
        video = torch.stack(frames).permute(3, 0, 1, 2).float() / 255.0
        video = F.interpolate(video.permute(1, 0, 2, 3), size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        return {"video": video, "label": label, "template": template, "video_id": path.stem}


class VideoDataset(Dataset):
    """Intentional extension point for a manifest or directory-backed video dataset."""
    def __init__(self, root: str | Path, **_):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        raise NotImplementedError("Implement decoding for your dataset format, or use SyntheticVideoDataset first.")
