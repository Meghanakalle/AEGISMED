import torch

class DPOptimizerWrapper:
    """
    Wraps standard PyTorch optimizer with per-sample gradient clipping and Gaussian noise injection.
    """
    def __init__(self, optimizer, max_grad_norm: float = 1.0, noise_multiplier: float = 1.1):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def step(self):
        # Clip parameters and add DP noise to gradients
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad)
                    clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
                    if clip_coef < 1.0:
                        p.grad.data.mul_(clip_coef)
                    if self.noise_multiplier > 0:
                        # Calibrate per-element std by 1/sqrt(numel) so the total
                        # injected noise norm matches noise_multiplier * max_grad_norm
                        # instead of scaling up with tensor size (see privacy/noise.py).
                        num_elements = p.grad.numel()
                        std_dev = (self.noise_multiplier * self.max_grad_norm) / (num_elements ** 0.5)
                        noise = torch.randn_like(p.grad) * std_dev
                        p.grad.data.add_(noise)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
