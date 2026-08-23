from utils.config import load_config
from data.loader import get_hospital_dataloaders

def test_data_loader_building():
    config = load_config("configs/fast_validation.yaml")
    hospital_loaders, global_test_loader, stats = get_hospital_dataloaders(config)
    
    assert len(hospital_loaders) == 4
    assert len(stats) == 4
    assert len(global_test_loader) > 0

    x_sample, y_sample = next(iter(hospital_loaders[0]))
    assert x_sample.shape[1] == 3  # RGB channels
    assert len(y_sample) == x_sample.shape[0]
