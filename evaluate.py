import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from fivek_project.dataset import FiveKPairDataset
from fivek_project.metrics import psnr, ssim
from fivek_project.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FiveK enhancement checkpoint.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--num-visuals", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    out_dir = Path(args.out_dir)
    comparison_dir = out_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    dataset = FiveKPairDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    stats = evaluate_model(model, loader, device)
    print(f"{args.split}_psnr={stats['psnr']:.2f}")
    print(f"{args.split}_ssim={stats['ssim']:.4f}")

    save_visual_comparisons(model, dataset, comparison_dir, args.num_visuals, device)
    print(f"Saved visual comparisons to {comparison_dir}")


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0}

    for inputs, targets in tqdm(loader, desc="evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs).clamp(0, 1)

        batch_size = inputs.size(0)
        totals["psnr"] += psnr(outputs, targets).item() * batch_size
        totals["ssim"] += ssim(outputs, targets).item() * batch_size

    return {key: value / len(loader.dataset) for key, value in totals.items()}


@torch.no_grad()
def save_visual_comparisons(
    model: nn.Module,
    dataset: FiveKPairDataset,
    out_dir: Path,
    num_visuals: int,
    device: torch.device,
) -> None:
    model.eval()
    count = min(num_visuals, len(dataset))

    for index in range(count):
        input_tensor, target_tensor = dataset[index]
        output_tensor = model(input_tensor.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)

        images = [
            tensor_to_pil(input_tensor),
            tensor_to_pil(output_tensor),
            tensor_to_pil(target_tensor),
        ]
        grid = make_labeled_grid(images, ["Original", "Model output", "Expert D target"])
        grid.save(out_dir / f"comparison_{index:03d}.jpg", quality=95)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return F.to_pil_image(tensor.cpu().clamp(0, 1))


def make_labeled_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    label_height = 28
    grid = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(grid)

    for column, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = column * width
        grid.paste(image, (x, label_height))
        draw.text((x + 8, 7), label, fill=(0, 0, 0))

    return grid


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
