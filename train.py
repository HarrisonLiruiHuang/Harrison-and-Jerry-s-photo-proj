import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fivek_project.dataset import FiveKPairDataset
from fivek_project.model import build_model
from fivek_project.metrics import psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained U-Net for FiveK enhancement.")
    parser.add_argument("--data-dir", default="data", help="Prepared data folder.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssim-weight", type=float, default=0.0, help="Optional weight for 1 - SSIM loss.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-pretrained", action="store_true", help="Train without ImageNet encoder weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        FiveKPairDataset(args.data_dir, "train", augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FiveKPairDataset(args.data_dir, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    l1_loss = nn.L1Loss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, l1_loss, device, epoch, args.ssim_weight)
        val_stats = evaluate(model, val_loader, l1_loss, device, args.ssim_weight)

        print(
            f"epoch {epoch:03d} | "
            f"train_l1={train_loss:.4f} | "
            f"val_l1={val_stats['loss']:.4f} | "
            f"val_psnr={val_stats['psnr']:.2f} | "
            f"val_ssim={val_stats['ssim']:.4f}"
        )

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, val_stats, args)
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, val_stats, args)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l1_loss: nn.Module,
    device: torch.device,
    epoch: int,
    ssim_weight: float,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"epoch {epoch} train")
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = enhancement_loss(outputs, targets, l1_loss, ssim_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(l1=loss.item())

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    l1_loss: nn.Module,
    device: torch.device,
    ssim_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs).clamp(0, 1)

        batch_size = inputs.size(0)
        totals["loss"] += enhancement_loss(outputs, targets, l1_loss, ssim_weight).item() * batch_size
        totals["psnr"] += psnr(outputs, targets).item() * batch_size
        totals["ssim"] += ssim(outputs, targets).item() * batch_size

    return {key: value / len(loader.dataset) for key, value in totals.items()}


def enhancement_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    l1_loss: nn.Module,
    ssim_weight: float,
) -> torch.Tensor:
    loss = l1_loss(outputs, targets)
    if ssim_weight > 0:
        loss = loss + ssim_weight * (1.0 - ssim(outputs, targets))
    return loss


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_stats: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_stats": val_stats,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
