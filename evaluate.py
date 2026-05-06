import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fivek_project.dataset import FiveKSuggestionDataset
from fivek_project.suggestion_model import build_suggestion_model
from fivek_project.suggestions import suggestions_from_labels, tensor_to_labels
from train import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-to-text editing suggestions.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/suggestions/best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    dataset = FiveKSuggestionDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_suggestion_model(pretrained=False).to(device)
    saved = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    model.eval()

    loss = evaluate_loss(model, loader, device)
    print(f"{args.split}_suggestion_smooth_l1={loss:.4f}")
    print_examples(model, dataset, args.examples, device)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    loss_fn = nn.SmoothL1Loss()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs)
        total_loss += loss_fn(predictions, labels).item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def print_examples(model: nn.Module, dataset: FiveKSuggestionDataset, examples: int, device: torch.device) -> None:
    for index in range(min(examples, len(dataset))):
        image_tensor, _ = dataset[index]
        prediction = model(image_tensor.unsqueeze(0).to(device)).squeeze(0)
        labels = tensor_to_labels(prediction)
        print(f"\nExample {index + 1}")
        for suggestion in suggestions_from_labels(labels):
            print(f"- {suggestion}")


if __name__ == "__main__":
    main()
