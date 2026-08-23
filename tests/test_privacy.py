import torch
from privacy.noise import apply_differential_privacy_noise
from privacy.privacy_accountant import RDPPrivacyAccountant

def test_dp_noise_and_accountant():
    tensor = torch.ones((10, 10))
    noisy_tensor = apply_differential_privacy_noise(tensor, noise_multiplier=1.0, max_grad_norm=1.0)
    
    assert noisy_tensor.shape == tensor.shape
    assert not torch.equal(tensor, noisy_tensor)

    accountant = RDPPrivacyAccountant(target_delta=1e-4)
    accountant.step(noise_multiplier=1.1, sample_rate=0.25, steps=10)
    eps = accountant.get_epsilon()
    assert eps > 0.0
