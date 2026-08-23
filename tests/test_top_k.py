import torch
from compression.top_k import apply_top_k_sparsification

def test_top_k_sparsification_ratio():
    state_dict = {"weight": torch.arange(100, dtype=torch.float32).reshape(10, 10)}
    compressed = apply_top_k_sparsification(state_dict, k_ratio=0.20)

    nonzero_count = torch.count_nonzero(compressed["weight"]).item()
    assert nonzero_count == 20
    assert compressed["weight"].shape == (10, 10)

    # The 20 largest-magnitude elements (80..99) should be preserved
    preserved = compressed["weight"].flatten()
    kept_values = set(preserved[preserved != 0].tolist())
    assert kept_values == set(float(v) for v in range(80, 100))

def test_top_k_disabled_passthrough():
    state_dict = {"weight": torch.ones((4, 4))}
    result = apply_top_k_sparsification(state_dict, k_ratio=1.0)
    torch.testing.assert_close(result["weight"], state_dict["weight"])
