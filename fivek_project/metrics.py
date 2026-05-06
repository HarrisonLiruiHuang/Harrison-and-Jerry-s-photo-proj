import torch
from torch.nn import functional as F


def psnr(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(output, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim(output: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = output.size(1)
    window = torch.ones(channels, 1, window_size, window_size, device=output.device, dtype=output.dtype)
    window = window / float(window_size * window_size)
    padding = window_size // 2

    mu_output = F.conv2d(output, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_output_sq = mu_output.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_output_target = mu_output * mu_target

    sigma_output_sq = F.conv2d(output * output, window, padding=padding, groups=channels) - mu_output_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_output_target = F.conv2d(output * target, window, padding=padding, groups=channels) - mu_output_target

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_output_target + c1) * (2 * sigma_output_target + c2)) / (
        (mu_output_sq + mu_target_sq + c1) * (sigma_output_sq + sigma_target_sq + c2)
    )
    return score.mean()
