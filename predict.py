import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from tqdm import tqdm

from fivek_project.model import build_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance new photos with a trained FiveK model.")
    parser.add_argument("--input", required=True, help="Input image file or folder of images.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out-dir", default="outputs/predictions")
    parser.add_argument("--image-size", type=int, default=256, help="Model input/output size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_paths = collect_images(input_path)
    if not image_paths:
        raise ValueError(f"No supported image files found at {input_path}")

    for image_path in tqdm(image_paths, desc="enhancing"):
        output = enhance_image(model, image_path, args.image_size, device)
        output.save(out_dir / f"{image_path.stem}_enhanced.jpg", quality=95)

    print(f"Saved {len(image_paths)} enhanced image(s) to {out_dir}")


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
def enhance_image(model: torch.nn.Module, image_path: Path, image_size: int, device: torch.device) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)

    tensor = F.to_tensor(image).unsqueeze(0).to(device)
    output = model(tensor).squeeze(0).cpu().clamp(0, 1)
    return F.to_pil_image(output)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
