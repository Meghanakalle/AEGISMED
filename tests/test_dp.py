import torch
from privacy.privacy_accountant import RDPPrivacyAccountant
from privacy.noise import apply_differential_privacy_noise

def test_privacy_accountant_and_noise():
    accountant = RDPPrivacyAccountant(target_delta=1e-4)
    accountant.step(noise_multiplier=1.1, sample_rate=0.25, steps=10)
    eps = accountant.get_epsilon()
    assert 0.0 < eps <= 15.0
    assert accountant.target_delta == 1e-4

    tensor = torch.zeros((5, 5))
    noisy_tensor = apply_differential_privacy_noise(tensor, noise_multiplier=1.1, max_grad_norm=1.0)
    assert not torch.equal(tensor, noisy_tensor)
