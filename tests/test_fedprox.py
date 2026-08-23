import torch
from federated.fedprox import fedprox_aggregate, compute_fedprox_proximal_term
from models.classifier import build_model
from utils.config import load_config

def test_fedprox_weight_aggregation():
    w1 = {"layer.weight": torch.ones((2, 2), dtype=torch.float32) * 1.0}
    w2 = {"layer.weight": torch.ones((2, 2), dtype=torch.float32) * 3.0}
    aggregated = fedprox_aggregate([w1, w2], [100, 100])
    torch.testing.assert_close(aggregated["layer.weight"], torch.ones((2, 2)) * 2.0)

def test_fedprox_proximal_term():
    config = load_config("configs/fast_validation.yaml")
    model = build_model(config)
    global_weights = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Identical weights -> zero proximal term
    zero_term = compute_fedprox_proximal_term(model, global_weights, mu=0.01)
    assert torch.isclose(zero_term, torch.tensor(0.0), atol=1e-5)

    # Perturbed weights -> positive proximal term
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)
    positive_term = compute_fedprox_proximal_term(model, global_weights, mu=0.01)
    assert positive_term.item() > 0.0
