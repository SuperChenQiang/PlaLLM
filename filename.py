import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import utils


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extension-free sample lists for 300W-LP / AFLW2000."
    )
    parser.add_argument(
        "--root_dir",
        "--root-dir",
        required=True,
        type=Path,
        help="Dataset root containing paired .jpg / .mat files.",
    )
    parser.add_argument(
        "--filename",
        default=Path("files.txt"),
        type=Path,
        help="Output path; relative paths are resolved under the dataset root.",
    )
    return parser.parse_args(argv)


def generate_filename_list(
    root_dir: str | Path, filename: str | Path = "files.txt"
) -> tuple[int, int, int]:
    """Write sorted relative stems; return (listed, out_of_range, missing_mat).

    Missing annotations are counted and skipped. Malformed annotations raise an
    error before an existing output file is opened. All angles must be within
    [-99, 99] degrees, matching the original helper.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    output = Path(filename).expanduser()
    if not output.is_absolute():
        output = root / output

    samples = []
    out_of_range = missing_mat = 0
    for image in sorted(root.rglob("*.jpg")):
        if not image.is_file():
            continue
        annotation = image.with_suffix(".mat")
        if not annotation.is_file():
            missing_mat += 1
            continue
        try:
            angles = np.rad2deg(utils.get_ypr_from_mat(annotation))
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            raise ValueError(
                f"Cannot read pose annotation {annotation}: {exc}"
            ) from exc
        if np.all(np.abs(angles) <= 99):
            samples.append(image.relative_to(root).with_suffix("").as_posix())
        else:
            out_of_range += 1

    # Resolve and validate all input annotations before replacing the output.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{sample}\n" for sample in samples), encoding="utf-8")
    return len(samples), out_of_range, missing_mat


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    listed, rejected, missing = generate_filename_list(args.root_dir, args.filename)
    print(
        f"{listed} files listed; {rejected} out-of-range poses excluded; "
        f"{missing} images skipped because .mat annotations were missing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
