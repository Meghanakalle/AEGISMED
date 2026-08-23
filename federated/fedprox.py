import torch

def compute_fedprox_proximal_term(model, global_model_weights_device, mu: float = 0.01) -> torch.Tensor:
    """
    Computes FedProx proximal regularization term: (mu / 2) * || w - w_global ||^2
    to handle statistical non-IID drift across hospitals.
    Expects global_model_weights_device to be a pre-flattened tensor for performance,
    or a state_dict mapping names to device-resident tensors for backward compatibility.
    """
    if mu <= 0.0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    if isinstance(global_model_weights_device, torch.Tensor):
        client_flat = torch.cat([
            param.flatten()
            for param in model.parameters()
            if param.requires_grad
        ])
        return (mu / 2.0) * torch.sum((client_flat - global_model_weights_device) ** 2)

    proximal_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_model_weights_device:
            global_weight = global_model_weights_device[name]
            proximal_loss += torch.sum((param - global_weight.to(param.device)) ** 2)

    return (mu / 2.0) * proximal_loss


def fedprox_aggregate(client_weights: list, client_sample_counts: list) -> dict:
    """
    Weighted FedAvg/FedProx aggregation of hospital client models based on dataset sample volume.
    """
    total_samples = sum(client_sample_counts)
    aggregated_weights = {}

    first_dict = client_weights[0]
    for key in first_dict.keys():
        if first_dict[key].is_floating_point():
            weighted_sum = torch.zeros_like(first_dict[key], dtype=torch.float32)
            for weights, count in zip(client_weights, client_sample_counts):
                weight = count / total_samples
                weighted_sum += weights[key].to(torch.float32) * weight
            aggregated_weights[key] = weighted_sum
        else:
            aggregated_weights[key] = first_dict[key].clone()

    return aggregated_weights
