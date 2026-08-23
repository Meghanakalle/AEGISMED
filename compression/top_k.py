import torch

# Tensor name fragments that must never be sparsified. These are normalization
# running statistics (BatchNorm/GroupNorm/InstanceNorm buffers), not learned
# weights. Zeroing 70-80% of running_var (as top-k with a 20-30% keep ratio
# does) drives many channels' variance estimate near zero, which blows up
# BatchNorm's 1/sqrt(var + eps) normalization and explodes activations/loss.
# Compressing these buffers also saves negligible bandwidth since they're tiny
# compared to conv/linear weight tensors, so excluding them costs nothing.
_NEVER_SPARSIFY_SUBSTRINGS = ("running_mean", "running_var", "num_batches_tracked")

def apply_top_k_sparsification(model_state_dict: dict, k_ratio: float = 0.20) -> dict:
    """
    Compresses model weight updates by retaining only top-k percentage tensors with largest absolute magnitude.
    Sets all remaining values to zero (sparsification).
    Normalization running statistics (BatchNorm etc.) are always kept dense - see
    _NEVER_SPARSIFY_SUBSTRINGS above.
    """
    if k_ratio >= 1.0 or k_ratio <= 0.0:
        return model_state_dict

    compressed_dict = {}
    for key, tensor in model_state_dict.items():
        if not tensor.is_floating_point() or any(s in key for s in _NEVER_SPARSIFY_SUBSTRINGS):
            compressed_dict[key] = tensor.clone()
            continue

        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()
        k = max(1, int(num_elements * k_ratio))

        if k >= num_elements:
            compressed_dict[key] = tensor.clone()
            continue

        abs_flat = torch.abs(flat_tensor)
        topk_vals, topk_indices = torch.topk(abs_flat, k)

        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[topk_indices] = True

        sparse_flat = torch.where(mask, flat_tensor, torch.zeros_like(flat_tensor))
        compressed_dict[key] = sparse_flat.view_as(tensor)

    return compressed_dict
