import numpy as np
from data.partitioner import partition_dirichlet

def test_dirichlet_partitioning():
    np.random.seed(0)
    labels = [i % 3 for i in range(40)]
    client_splits = partition_dirichlet(labels, num_clients=4, alpha=0.3, num_classes=3, seed=42)

    assert len(client_splits) == 4
    total_partitioned = sum(len(s) for s in client_splits)
    assert total_partitioned == 40

    # Every original index should appear exactly once across all clients
    all_indices = sorted([idx for split in client_splits for idx in split])
    assert all_indices == list(range(40))
