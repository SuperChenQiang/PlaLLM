"""Regression checks for pose targets, file formats, and the filename CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

import datasets
import filename
import utils


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "dataset with spaces"
    root.mkdir()
    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    pixels[:, :50, 0] = 255
    pixels[:, 50:, 2] = 255
    Image.fromarray(pixels).save(root / "face.jpg")
    savemat(
        root / "face.mat",
        {
            "Pose_Para": np.deg2rad([[10, 20, 30, 0, 0, 0, 1]]),
            "pt2d": np.array([[25, 75, 25, 75], [20, 20, 60, 60]]),
        },
    )
    listing = root / "files.txt"
    listing.write_text("\ufeff\nface\n  \n", encoding="utf-8")
    return root, listing


def no_random():
    raise AssertionError("Evaluation must not use random augmentation")


def test_rotation_convention():
    angles = [0.31, -0.62, 0.93]
    actual = utils.get_R(*angles)
    expected = Rotation.from_euler("xyz", angles).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-14)
    assert np.linalg.det(actual) == pytest.approx(1)


def test_mat_row_and_column_vectors(tmp_path):
    for shape in [(1, 7), (7, 1)]:
        path = tmp_path / f"{shape[0]}.mat"
        savemat(path, {"Pose_Para": np.arange(7).reshape(shape)})
        np.testing.assert_array_equal(utils.get_ypr_from_mat(path), [0, 1, 2])


def test_aflw2000_preserves_radian_order_and_reads_mat_once(sample, monkeypatch):
    root, listing = sample
    calls = []
    real_loadmat = utils.loadmat

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real_loadmat(*args, **kwargs)

    monkeypatch.setattr(utils, "loadmat", counted)
    image, rotation, angles, name = datasets.AFLW2000(root, listing, None)[0]
    np.testing.assert_allclose(angles, np.deg2rad([20, 10, 30]), rtol=1e-6)
    np.testing.assert_allclose(
        rotation,
        Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(),
        rtol=1e-6,
    )
    assert len(calls) == 1
    assert name == "face"
    assert image.size == (98, 62)  # Legacy crop rule.
    assert rotation.dtype == torch.float32


def test_300w_flip_updates_both_targets_and_image(sample, monkeypatch):
    root, listing = sample
    draws = iter([0.5, 0.1, 0.9])  # Crop, flip, no blur.
    monkeypatch.setattr(np.random, "random_sample", lambda: next(draws))
    flipped, rotation, angles, _ = datasets.Pose_300W_LP(root, listing, None)[0]
    np.testing.assert_allclose(angles, [10, -20, -30], atol=1e-5)
    np.testing.assert_allclose(
        rotation,
        Rotation.from_euler("xyz", [10, -20, -30], degrees=True).as_matrix(),
        rtol=1e-6,
    )
    unflipped = datasets.Pose_300W_LP(root, listing, None, train_mode=False)[0][0]
    np.testing.assert_array_equal(
        np.asarray(flipped),
        np.asarray(unflipped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
    )


def test_300w_eval_is_deterministic_through_factory(sample, monkeypatch):
    root, listing = sample
    monkeypatch.setattr(np.random, "random_sample", no_random)
    data = datasets.getDataset("Pose_300W_LP", root, listing, None, train_mode=False)
    first, second = data[0], data[0]
    np.testing.assert_array_equal(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    np.testing.assert_allclose(first[2], [10, 20, 30], atol=1e-5)


def test_biwi_flip_and_legacy_output(tmp_path, monkeypatch):
    images = np.zeros((1, 8, 10, 3), dtype=np.uint8)
    images[:, :, :5, 0] = 255
    path = tmp_path / "biwi.npz"
    np.savez(path, image=images, pose=[[20, 10, 30]])
    draws = iter([0.1, 0.9])
    monkeypatch.setattr(np.random, "random_sample", lambda: next(draws))
    image, rotation, angles, original = datasets.BIWI(tmp_path, path, None)[0]
    np.testing.assert_allclose(angles, np.deg2rad([10, -20, -30]), rtol=1e-6)
    np.testing.assert_allclose(
        rotation,
        Rotation.from_euler("xyz", [10, -20, -30], degrees=True).as_matrix(),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(original, images[0])
    np.testing.assert_array_equal(np.asarray(image), images[0, :, ::-1])
    monkeypatch.setattr(np.random, "random_sample", no_random)
    evaluation = datasets.getDataset("BIWI", tmp_path, path, None, train_mode=False)
    np.testing.assert_allclose(evaluation[0][2], np.deg2rad([10, 20, 30]), rtol=1e-6)
    # The context-managed archive is closed; the materialized arrays still work.
    path.unlink()
    assert evaluation[0][0].size == (10, 8)


@pytest.mark.parametrize(
    "archive, message",
    [
        ({"image": np.zeros((1, 8, 10, 3))}, "requires"),
        ({"image": np.zeros((2, 8, 10, 3)), "pose": [[0, 0, 0]]}, "matching"),
        ({"image": np.zeros((1, 8, 10, 3)), "pose": [[0, np.nan, 0]]}, "finite"),
        ({"image": np.zeros((1, 8, 10, 1)), "pose": [[0, 0, 0]]}, "channels"),
    ],
)
def test_biwi_rejects_invalid_archives(tmp_path, archive, message):
    path = tmp_path / "bad.npz"
    np.savez(path, **archive)
    with pytest.raises(ValueError, match=message):
        datasets.BIWI(tmp_path, path, None)


def test_aflw_whitespace_and_roll_convention(sample):
    root, listing = sample
    (root / "face.txt").write_text("face\t0.1   0.2\t0.3\n")
    _, bins, angles, _ = datasets.AFLW(root, listing, None)[0]
    np.testing.assert_allclose(angles, np.rad2deg([0.1, 0.2, -0.3]), rtol=1e-6)
    assert bins.dtype == torch.int64


def test_afw_preserves_directory_and_image_underscores(tmp_path):
    folder = tmp_path / "folder_with_underscores"
    folder.mkdir()
    Image.new("RGB", (100, 80)).save(folder / "face_name.jpg")
    (folder / "face_name_0.txt").write_text("face_name 10 20 30 25 20 75 60\n")
    listing = tmp_path / "files.txt"
    listing.write_text("folder_with_underscores/face_name_0\n")
    _, _, angles, name = datasets.AFW(tmp_path, listing, None)[0]
    np.testing.assert_array_equal(angles, [10, 20, 30])
    assert name == "folder_with_underscores/face_name_0"


def test_filename_sorted_filtering_and_cli(sample):
    root, _ = sample
    for name, angle in [("z", 99), ("a", -99), ("excluded", 100)]:
        Image.new("RGB", (8, 8)).save(root / f"{name}.jpg")
        savemat(root / f"{name}.mat", {"Pose_Para": np.deg2rad([[angle, 0, 0]])})
    Image.new("RGB", (8, 8)).save(root / "missing.jpg")
    cwd = Path.cwd()
    assert filename.generate_filename_list(root, "generated.txt") == (3, 1, 1)
    assert Path.cwd() == cwd
    assert (root / "generated.txt").read_text().splitlines() == ["a", "face", "z"]
    result = subprocess.run(
        [
            sys.executable,
            str(Path(filename.__file__)),
            "--root_dir",
            str(root),
            "--filename",
            "cli.txt",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "3 files listed" in result.stdout
    assert (root / "cli.txt").read_bytes() == (root / "generated.txt").read_bytes()
    assert filename.parse_args(["--root-dir", str(root)]).root_dir == root


def test_bad_annotations_do_not_truncate_existing_list(sample):
    root, listing = sample
    original = listing.read_bytes()
    savemat(root / "face.mat", {"Pose_Para": [[np.nan, 0, 0]]})
    with pytest.raises(ValueError, match="face.mat"):
        filename.generate_filename_list(root)
    assert listing.read_bytes() == original


def test_missing_root_and_unknown_loader(tmp_path):
    with pytest.raises(NotADirectoryError):
        filename.generate_filename_list(tmp_path / "missing")
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.getDataset("unknown", tmp_path, tmp_path / "list.txt", None)


def test_malformed_landmarks_have_actionable_error(sample):
    root, listing = sample
    savemat(root / "face.mat", {"Pose_Para": [[0, 0, 0]], "pt2d": [[1], [2]]})
    with pytest.raises(ValueError, match="nonzero area"):
        datasets.AFLW2000(root, listing, None)[0]


def test_torch_dataloader_batches_eval_samples(sample):
    root, listing = sample

    def to_tensor(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)

    data = datasets.getDataset(
        "Pose_300W_LP", root, listing, to_tensor, train_mode=False
    )
    image, rotation, angles, names = next(iter(DataLoader(data, batch_size=1)))
    assert image.ndim == 4
    assert rotation.shape == (1, 3, 3)
    assert angles.shape == (1, 3)
    assert names == ("face",)
