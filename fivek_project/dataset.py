from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from fivek_project.suggestions import labels_from_pair, labels_to_tensor


Split = Literal["train", "val", "test"]


class FiveKPairDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: Split = "train", augment: bool = False):
        self.input_dir = Path(data_dir) / split / "input"
        self.target_dir = Path(data_dir) / split / "target"
        self.augment = augment and split == "train"

        self.input_paths = sorted(self.input_dir.glob("*"))
        target_names = {path.name for path in self.target_dir.glob("*")}
        self.input_paths = [path for path in self.input_paths if path.name in target_names]

        if not self.input_paths:
            raise ValueError(f"No paired images found under {self.input_dir} and {self.target_dir}")

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.input_paths[index]
        target_path = self.target_dir / input_path.name

        input_image = _open_rgb(input_path)
        target_image = _open_rgb(target_path)

        if self.augment and torch.rand(()) < 0.5:
            input_image = F.hflip(input_image)
            target_image = F.hflip(target_image)

        return F.to_tensor(input_image), F.to_tensor(target_image)


class FiveKSuggestionDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: Split = "train", augment: bool = False):
        self.input_dir = Path(data_dir) / split / "input"
        self.target_dir = Path(data_dir) / split / "target"
        self.augment = augment and split == "train"

        self.input_paths = sorted(self.input_dir.glob("*"))
        target_names = {path.name for path in self.target_dir.glob("*")}
        self.input_paths = [path for path in self.input_paths if path.name in target_names]

        if not self.input_paths:
            raise ValueError(f"No paired images found under {self.input_dir} and {self.target_dir}")

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.input_paths[index]
        target_path = self.target_dir / input_path.name

        input_image = _open_rgb(input_path)
        target_image = _open_rgb(target_path)

        if self.augment and torch.rand(()) < 0.5:
            input_image = F.hflip(input_image)
            target_image = F.hflip(target_image)

        labels = labels_to_tensor(labels_from_pair(input_image, target_image))
        return F.to_tensor(input_image), labels


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
