import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

import utils

PathValue = str | os.PathLike[str]
Transform = Callable[[Image.Image], object] | None
ANGLE_BINS = np.arange(-99, 102, 3)


def get_list_from_filenames(file_path: PathValue) -> list[str]:
    """Read relative sample stems, ignoring blank lines and a UTF-8 BOM."""
    with open(file_path, encoding="utf-8-sig") as stream:
        return [line.strip().replace("\\", "/") for line in stream if line.strip()]


def _read_image(path: PathValue, mode: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert(mode)


def _read_annotation(path: PathValue, count: int) -> np.ndarray:
    with open(path, encoding="utf-8-sig") as stream:
        fields = stream.readline().split()
    try:
        values = np.asarray(fields[1 : count + 1], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: annotation must contain numeric values") from exc
    if values.size != count or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected {count} finite values after the image name")
    return values


def _augment(
    image: Image.Image, pitch: float, yaw: float, roll: float
) -> tuple[Image.Image, float, float, float]:
    if np.random.random_sample() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        yaw, roll = -yaw, -roll
    if np.random.random_sample() < 0.05:
        image = image.filter(ImageFilter.BLUR)
    return image, pitch, yaw, roll


class _FileDataset(Dataset):
    """Shared file-list setup for image/annotation pairs."""

    default_annot_ext = ".mat"

    def __init__(
        self,
        data_dir: PathValue,
        filename_path: PathValue,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str | None = None,
        image_mode: str = "RGB",
    ) -> None:
        self.data_dir = os.fspath(data_dir)
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = self.default_annot_ext if annot_ext is None else annot_ext
        self.X_train = get_list_from_filenames(filename_path)
        self.y_train = self.X_train
        self.image_mode = image_mode
        self.length = len(self.X_train)

    def __len__(self) -> int:
        return self.length


class AFLW2000(_FileDataset):
    """Return (image, rotation, [yaw, pitch, roll] in radians, sample stem)."""

    def __getitem__(self, index: int) -> tuple[object, torch.Tensor, torch.Tensor, str]:
        sample = self.X_train[index]
        image = _read_image(
            os.path.join(self.data_dir, sample + self.img_ext), self.image_mode
        )
        mat_path = os.path.join(self.data_dir, sample + self.annot_ext)
        points, (pitch, yaw, roll) = utils.load_pose_annotation(mat_path)

        x_min, y_min = points.min(axis=1)
        x_max, y_max = points.max(axis=1)
        # Preserve the original crop expansion order for benchmark compatibility.
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        rotation = torch.as_tensor(utils.get_R(pitch, yaw, roll), dtype=torch.float32)
        labels = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, rotation, labels, sample


class AFLW(_FileDataset):
    """Return (image, bins, [yaw, pitch, roll] in degrees, sample stem)."""

    default_annot_ext = ".txt"

    def __getitem__(self, index: int) -> tuple[object, torch.Tensor, torch.Tensor, str]:
        sample = self.X_train[index]
        image = _read_image(
            os.path.join(self.data_dir, sample + self.img_ext), self.image_mode
        )
        angles = np.rad2deg(
            _read_annotation(os.path.join(self.data_dir, sample + self.annot_ext), 3)
        )
        angles[2] *= -1  # AFLW roll convention.
        labels = torch.as_tensor(np.digitize(angles, ANGLE_BINS) - 1, dtype=torch.int64)
        continuous = torch.as_tensor(angles, dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, labels, continuous, sample


class AFW(_FileDataset):
    """Return (image, bins, [yaw, pitch, roll] in degrees, sample stem)."""

    default_annot_ext = ".txt"

    def __getitem__(self, index: int) -> tuple[object, torch.Tensor, torch.Tensor, str]:
        sample = self.X_train[index]
        # Only strip the final face index, retaining directories/name underscores.
        sample_path = Path(sample)
        image_stem = sample_path.name.rsplit("_", 1)[0]
        image_path = (
            Path(self.data_dir) / sample_path.parent / (image_stem + self.img_ext)
        )
        image = _read_image(image_path, self.image_mode)
        values = _read_annotation(
            os.path.join(self.data_dir, sample + self.annot_ext), 7
        )
        yaw, pitch, roll, x1, y1, x2, y2 = values

        k = 0.32
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += k * abs(y2 - y1)
        image = image.crop((int(x1), int(y1), int(x2), int(y2)))

        angles = [yaw, pitch, roll]
        labels = torch.as_tensor(np.digitize(angles, ANGLE_BINS) - 1, dtype=torch.int64)
        continuous = torch.tensor(angles, dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, labels, continuous, sample


class BIWI(Dataset):
    """Read .npz poses in degrees, stored as [yaw, pitch, roll].

    Return (image, rotation, [pitch, yaw, roll] in radians, raw image array).
    The fourth item and output angle convention retain the repository's API.
    """

    def __init__(
        self,
        data_dir: PathValue,
        filename_path: PathValue,
        transform: Transform,
        image_mode: str = "RGB",
        train_mode: bool = True,
    ) -> None:
        self.data_dir = os.fspath(data_dir)
        self.transform = transform
        with np.load(filename_path, allow_pickle=False) as archive:
            if not {"image", "pose"}.issubset(archive.files):
                raise ValueError(
                    f"{filename_path}: BIWI archive requires 'image' and 'pose' arrays"
                )
            self.X_train = archive["image"]
            self.y_train = archive["pose"]
        if self.X_train.ndim not in (3, 4):
            raise ValueError(
                f"{filename_path}: BIWI image array must have shape (N, H, W[, C])"
            )
        if self.X_train.ndim == 4 and self.X_train.shape[-1] not in (3, 4):
            raise ValueError(
                f"{filename_path}: BIWI color images require 3 or 4 channels"
            )
        if self.y_train.shape != (len(self.X_train), 3):
            raise ValueError(
                f"{filename_path}: BIWI pose array must have shape (N, 3), matching images"
            )
        if not np.isfinite(self.y_train).all():
            raise ValueError(f"{filename_path}: BIWI poses must contain finite angles")
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(self.X_train)

    def __getitem__(
        self, index: int
    ) -> tuple[object, torch.Tensor, torch.Tensor, np.ndarray]:
        image = Image.fromarray(np.uint8(self.X_train[index])).convert(self.image_mode)
        yaw, pitch, roll = np.deg2rad(self.y_train[index])
        if self.train_mode:
            image, pitch, yaw, roll = _augment(image, pitch, yaw, roll)
        rotation = torch.as_tensor(utils.get_R(pitch, yaw, roll), dtype=torch.float32)
        continuous = torch.tensor([pitch, yaw, roll], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, rotation, continuous, self.X_train[index]

    def __len__(self) -> int:
        return self.length


class Pose_300W_LP(_FileDataset):
    """Return (image, rotation, [pitch, yaw, roll] in degrees, sample stem).

    Labels reflect horizontal flipping. Set train_mode=False for deterministic
    crops (k=0.3) and no random flip/blur, including through getDataset().
    """

    def __init__(
        self,
        data_dir: PathValue,
        filename_path: PathValue,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str = ".mat",
        image_mode: str = "RGB",
        train_mode: bool = True,
    ) -> None:
        super().__init__(
            data_dir, filename_path, transform, img_ext, annot_ext, image_mode
        )
        self.train_mode = train_mode

    def __getitem__(self, index: int) -> tuple[object, torch.Tensor, torch.Tensor, str]:
        sample = self.X_train[index]
        image = _read_image(
            os.path.join(self.data_dir, sample + self.img_ext), self.image_mode
        )
        mat_path = os.path.join(self.data_dir, sample + self.annot_ext)
        points, (pitch, yaw, roll) = utils.load_pose_annotation(mat_path)
        x_min, y_min = points.min(axis=1)
        x_max, y_max = points.max(axis=1)
        crop_fraction = np.random.random_sample() if self.train_mode else 0.5
        k = crop_fraction * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        if self.train_mode:
            image, pitch, yaw, roll = _augment(image, pitch, yaw, roll)

        rotation = torch.as_tensor(utils.get_R(pitch, yaw, roll), dtype=torch.float32)
        # Generate angles after augmentation so both targets describe this image.
        continuous = torch.as_tensor(
            np.rad2deg([pitch, yaw, roll]), dtype=torch.float32
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, rotation, continuous, sample


def getDataset(
    dataset: str,
    data_dir: PathValue,
    filename_list: PathValue,
    transformations: Transform,
    train_mode: bool = True,
) -> Dataset:
    """Build a loader using the existing public dataset names."""
    loaders = {
        "Pose_300W_LP": Pose_300W_LP,
        "AFLW2000": AFLW2000,
        "BIWI": BIWI,
        "AFLW": AFLW,
        "AFW": AFW,
    }
    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose from {', '.join(loaders)}"
        )
    options = {"train_mode": train_mode} if dataset in {"Pose_300W_LP", "BIWI"} else {}
    return loaders[dataset](data_dir, filename_list, transformations, **options)
