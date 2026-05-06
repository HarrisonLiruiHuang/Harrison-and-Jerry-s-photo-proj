from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision.transforms import functional as F


EDIT_NAMES = [
    "brightness",
    "contrast",
    "highlights",
    "shadows",
    "temperature",
    "tint",
    "saturation",
    "clarity",
]


@dataclass
class ImageStats:
    brightness: float
    contrast: float
    highlights: float
    shadows: float
    warmth: float
    tint: float
    saturation: float
    clarity: float


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return F.to_tensor(image)


def tensor_to_labels(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().tolist()
    return dict(zip(EDIT_NAMES, values, strict=True))


def labels_to_tensor(labels: dict[str, float]) -> torch.Tensor:
    return torch.tensor([labels[name] for name in EDIT_NAMES], dtype=torch.float32)


def labels_from_pair(input_image: Image.Image, target_image: Image.Image) -> dict[str, float]:
    before = compute_stats(input_image)
    after = compute_stats(target_image)
    return {
        "brightness": clamp((after.brightness - before.brightness) * 2.0),
        "contrast": clamp((after.contrast - before.contrast) * 2.4),
        "highlights": clamp((after.highlights - before.highlights) * 2.0),
        "shadows": clamp((after.shadows - before.shadows) * 2.0),
        "temperature": clamp((after.warmth - before.warmth) * 3.0),
        "tint": clamp((after.tint - before.tint) * 3.0),
        "saturation": clamp((after.saturation - before.saturation) * 2.2),
        "clarity": clamp((after.clarity - before.clarity) * 2.0),
    }


def compute_stats(image: Image.Image) -> ImageStats:
    image = image.convert("RGB")
    grayscale = image.convert("L")
    hsv = image.convert("HSV")
    gray_data = list(grayscale.getdata())
    rgb_stats = ImageStat.Stat(image)
    gray_stats = ImageStat.Stat(grayscale)
    sat_stats = ImageStat.Stat(hsv.getchannel("S"))

    brightness = gray_stats.mean[0] / 255.0
    contrast = gray_stats.stddev[0] / 255.0
    highlights = channel_mean([value for value in gray_data if value >= 200], default=gray_stats.mean[0]) / 255.0
    shadows = channel_mean([value for value in gray_data if value <= 60], default=gray_stats.mean[0]) / 255.0
    red, green, blue = [value / 255.0 for value in rgb_stats.mean]
    warmth = red - blue
    tint = ((red + blue) / 2.0) - green
    saturation = sat_stats.mean[0] / 255.0
    clarity = contrast

    return ImageStats(
        brightness=brightness,
        contrast=contrast,
        highlights=highlights,
        shadows=shadows,
        warmth=warmth,
        tint=tint,
        saturation=saturation,
        clarity=clarity,
    )


def suggestions_from_labels(labels: dict[str, float], image: Image.Image | None = None) -> list[str]:
    suggestions = [
        f"Brightness: {direction(labels['brightness'])}",
        f"Contrast: {direction(labels['contrast'])}",
        f"Highlights: {direction(labels['highlights'])}",
        f"Shadows: {direction(labels['shadows'])}",
        f"Temperature: {temperature_direction(labels['temperature'])}",
        f"Tint: {tint_direction(labels['tint'])}",
        f"Saturation: {direction(labels['saturation'])}",
        f"Clarity: {direction(labels['clarity'])}",
    ]

    if image is not None:
        stats = compute_stats(image)
        if stats.brightness < 0.34:
            suggestions.append("Noise reduction: consider light noise reduction in dark shadow areas.")
        else:
            suggestions.append("Noise reduction: no major noise reduction needed.")
        suggestions.append("Crop: no automatic crop recommendation; inspect subject placement manually.")

    suggestions.append(f"Overall suggestion: {overall_sentence(labels)}")
    return suggestions


def slider_defaults_from_labels(labels: dict[str, float]) -> dict[str, int]:
    return {
        "brightness": slider_value(labels["brightness"], scale=36),
        "contrast": slider_value(labels["contrast"], scale=34),
        "saturation": slider_value(labels["saturation"], scale=40),
        "temperature": slider_value(labels["temperature"], scale=24),
        "clarity": slider_value(labels["clarity"], scale=22),
    }


def direction(value: float) -> str:
    if value > 0.35:
        return "increase significantly"
    if value > 0.16:
        return "increase moderately"
    if value > 0.06:
        return "increase slightly"
    if value < -0.35:
        return "decrease significantly"
    if value < -0.16:
        return "decrease moderately"
    if value < -0.06:
        return "decrease slightly"
    return "keep about the same"


def temperature_direction(value: float) -> str:
    if value > 0.16:
        return "make warmer"
    if value > 0.06:
        return "make slightly warmer"
    if value < -0.16:
        return "make cooler"
    if value < -0.06:
        return "make slightly cooler"
    return "keep neutral"


def tint_direction(value: float) -> str:
    if value > 0.12:
        return "move slightly toward magenta"
    if value < -0.12:
        return "move slightly toward green"
    return "keep mostly neutral"


def overall_sentence(labels: dict[str, float]) -> str:
    phrases = []
    if labels["brightness"] > 0.08:
        phrases.append("make the image brighter")
    elif labels["brightness"] < -0.08:
        phrases.append("make the image darker")
    if labels["temperature"] > 0.08:
        phrases.append("warm the color tone")
    elif labels["temperature"] < -0.08:
        phrases.append("cool the color tone")
    if labels["contrast"] > 0.08:
        phrases.append("add contrast")
    elif labels["contrast"] < -0.08:
        phrases.append("soften contrast")
    if labels["saturation"] > 0.08:
        phrases.append("increase color vibrance")
    elif labels["saturation"] < -0.08:
        phrases.append("reduce color intensity")

    if not phrases:
        return "keep edits subtle and preserve the current natural look."
    return ", ".join(phrases) + "."


def slider_value(value: float, scale: int) -> int:
    return int(round(clamp(value) * scale))


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def channel_mean(values: list[int], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)
