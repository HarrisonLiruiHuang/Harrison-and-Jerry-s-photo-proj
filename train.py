import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fivek_project.dataset import FiveKSuggestionDataset
from fivek_project.suggestion_model import build_suggestion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-to-text photo editing suggestion model.")
    parser.add_argument("--data-dir", default="data", help="Prepared FiveK data folder.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints/suggestions")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        FiveKSuggestionDataset(args.data_dir, "train", augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FiveKSuggestionDataset(args.data_dir, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_suggestion_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, val_loss, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, val_loss, args)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"epoch {epoch} train")
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


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
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
