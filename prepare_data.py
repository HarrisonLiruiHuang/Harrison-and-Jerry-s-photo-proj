import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small paired FiveK subset.")
    parser.add_argument("--raw-dir", default="../raw", help="Folder with original unedited images.")
    parser.add_argument("--target-dir", default="../d", help="Folder with Expert D retouched images.")
    parser.add_argument("--out-dir", default="data", help="Output folder for train/val/test splits.")
    parser.add_argument("--limit", type=int, default=100, help="Number of paired images to use.")
    parser.add_argument("--image-size", type=int, default=256, help="Square RGB output size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    target_dir = Path(args.target_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw input folder not found: {raw_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Expert D target folder not found: {target_dir}")

    pairs = find_pairs(raw_dir, target_dir)[: args.limit]
    if not pairs:
        raise ValueError("No matching image filenames found between raw and target folders.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    splits = split_pairs(pairs)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split_name, split_pairs_for_name in splits.items():
        for side in ("input", "target"):
            (out_dir / split_name / side).mkdir(parents=True, exist_ok=True)

        for raw_path, target_path in tqdm(split_pairs_for_name, desc=f"Preparing {split_name}"):
            save_resized_rgb(raw_path, out_dir / split_name / "input" / raw_path.name, args.image_size)
            save_resized_rgb(target_path, out_dir / split_name / "target" / target_path.name, args.image_size)

    print(f"Prepared {len(pairs)} image pairs in {out_dir}")
    for split_name, split_pairs_for_name in splits.items():
        print(f"{split_name}: {len(split_pairs_for_name)} pairs")


def find_pairs(raw_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    raw_paths = {
        path.name: path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    target_paths = {
        path.name: path
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    return [(raw_paths[name], target_paths[name]) for name in sorted(raw_paths.keys() & target_paths.keys())]


def split_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    train_end = int(len(pairs) * 0.8)
    val_end = int(len(pairs) * 0.9)
    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }


def save_resized_rgb(src: Path, dst: Path, image_size: int) -> None:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)
        image.save(dst, quality=95)


if __name__ == "__main__":
    main()
