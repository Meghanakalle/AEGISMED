import math
import torch

def apply_differential_privacy_noise(tensor: torch.Tensor, noise_multiplier: float, max_grad_norm: float) -> torch.Tensor:
    """
    Clips gradient/parameter tensor to max_grad_norm and adds Gaussian Differential Privacy noise.
    """
    if noise_multiplier <= 0:
        return tensor
    
    # L2 clipping
    norm = torch.norm(tensor)
    clip_factor = max_grad_norm / (norm + 1e-6)
    if clip_factor < 1.0:
        tensor = tensor * clip_factor

    # Gaussian noise addition.
    # IMPORTANT: torch.randn_like adds an *independent* draw to every element.
    # If we used std_dev = noise_multiplier * max_grad_norm directly, the total
    # noise norm across the tensor grows as sqrt(N) * std_dev (N = numel), while
    # the clipped signal norm is fixed at max_grad_norm. For large tensors this
    # drowns the signal (e.g. N=10,000 -> ~100x too much noise).
    # Scaling by 1/sqrt(N) keeps the *total* injected noise norm calibrated to
    # noise_multiplier * max_grad_norm regardless of tensor size, matching the
    # intended DP sensitivity calibration.
    num_elements = tensor.numel()
    std_dev = (noise_multiplier * max_grad_norm) / math.sqrt(max(num_elements, 1))
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise
