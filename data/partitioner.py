import numpy as np

def partition_dirichlet(labels, num_clients: int = 4, alpha: float = 0.3, num_classes: int = 3, seed: int = 42):
    """
    Partitions dataset indices across num_clients according to a Dirichlet distribution (Dir(alpha)).
    Lower alpha generates higher statistical heterogeneity (Non-IID).
    """
    np.random.seed(seed)
    labels = np.array(labels)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize to split indices
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split_idx = np.split(idx_c, proportions)
        for i, idx_group in enumerate(split_idx):
            client_indices[i].extend(idx_group.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices
