import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from tqdm import tqdm

from fivek_project.suggestion_model import build_suggestion_model
from fivek_project.suggestions import suggestions_from_labels, tensor_to_labels


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate photo editing suggestions for new images.")
    parser.add_argument("--input", required=True, help="Input image file or folder of images.")
    parser.add_argument("--checkpoint", default="checkpoints/suggestions/best.pt")
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    model = load_model(args.checkpoint, device)

    image_paths = collect_images(Path(args.input))
    if not image_paths:
        raise ValueError(f"No supported image files found at {args.input}")

    for image_path in tqdm(image_paths, desc="suggesting"):
        image, labels, suggestions = suggest_for_image(model, image_path, args.image_size, device)
        print(f"\n{image_path}")
        for suggestion in suggestions:
            print(f"- {suggestion}")


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_suggestion_model(pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


@torch.no_grad()
def suggest_for_image(
    model: torch.nn.Module,
    image_path: str | Path,
    image_size: int,
    device: torch.device,
) -> tuple[Image.Image, dict[str, float], list[str]]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)

    tensor = F.to_tensor(image).unsqueeze(0).to(device)
    prediction = model(tensor).squeeze(0)
    labels = tensor_to_labels(prediction)
    suggestions = suggestions_from_labels(labels, image)
    return image, labels, suggestions


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
