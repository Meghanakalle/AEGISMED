from utils.config import load_config
from data.loader import get_hospital_dataloaders
from federated.server import FederatedServer

def test_federated_server_training_loop():
    config = load_config("configs/fast_validation.yaml")
    hospital_loaders, global_test_loader, _ = get_hospital_dataloaders(config)
    
    server = FederatedServer(
        config=config,
        hospital_loaders=hospital_loaders,
        global_test_loader=global_test_loader
    )
    df_results = server.run_training_loop(rounds=1)
    assert len(df_results) == 1
    assert "test_accuracy" in df_results.columns
    assert "communication_reduction_percent" in df_results.columns
