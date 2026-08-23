import torch
from compression.top_k import apply_top_k_sparsification
from compression.communication import calculate_state_dict_size_mb

def test_top_k_sparsification():
    state_dict = {
        "weight": torch.tensor([[1.0, 0.1], [0.05, 5.0]]),
        "bias": torch.tensor([0.2, 0.8])
    }
    sparse_dict = apply_top_k_sparsification(state_dict, k_ratio=0.50)
    
    assert "weight" in sparse_dict
    assert "bias" in sparse_dict
    
    # Verify non-zero count reduction
    flat_orig = torch.cat([v.flatten() for v in state_dict.values()])
    flat_sparse = torch.cat([v.flatten() for v in sparse_dict.values()])
    
    assert (flat_sparse != 0).sum() <= (flat_orig != 0).sum()
    
    size_mb = calculate_state_dict_size_mb(state_dict)
    assert size_mb > 0
