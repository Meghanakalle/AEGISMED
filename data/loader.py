import numpy as np
from torch.utils.data import DataLoader, Subset
from data.synthetic import generate_synthetic_medical_xray
from data.real_data import load_real_medical_dataset
from data.dataset import MedicalDataset
from data.preprocessing import get_transforms
from data.partitioner import partition_dirichlet

def get_hospital_dataloaders(config: dict, return_val: bool = False):
    dataset_cfg = config["dataset"]
    fed_cfg = config["federated"]

    image_size = dataset_cfg.get("image_size", 224)
    num_classes = dataset_cfg.get("num_classes", 3)
    num_hospitals = fed_cfg.get("num_hospitals", 4)
    alpha = dataset_cfg.get("dirichlet_alpha", 0.3)
    batch_size = fed_cfg.get("batch_size", 16)

    np.random.seed(config.get("seed", 42))

    data_dir = dataset_cfg.get("data_dir")
    if data_dir:
        # Real dataset: expects one subfolder per class under data_dir
        classes = dataset_cfg.get("classes", list(range(num_classes)))
        max_samples_per_class = dataset_cfg.get("max_samples_per_class")
        images, labels = load_real_medical_dataset(data_dir, classes, max_samples_per_class=max_samples_per_class)
        num_samples = len(images)
    else:
        # Synthetic fallback: generate procedurally-textured X-ray-like images
        num_samples = dataset_cfg.get("synthetic_samples", 800)
        labels = np.random.choice(num_classes, size=num_samples).tolist()
        images = [generate_synthetic_medical_xray(lbl, image_size=image_size) for lbl in labels]

    train_transform = get_transforms(image_size=image_size, is_train=True)
    test_transform = get_transforms(image_size=image_size, is_train=False)

    full_dataset = MedicalDataset(images, labels)

    # 80/20 train/test split
    num_train = int(0.8 * num_samples)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:num_train], indices[num_train:]

    if return_val:
        # Split train_indices into 90% training and 10% validation
        num_train_train = int(0.9 * len(train_indices))
        train_train_indices = train_indices[:num_train_train]
        val_indices = train_indices[num_train_train:]
    else:
        train_train_indices = train_indices
        val_indices = []

    train_labels = [labels[i] for i in train_train_indices]
    
    # Non-IID Dirichlet partition on training set
    client_train_splits = partition_dirichlet(train_labels, num_clients=num_hospitals, alpha=alpha, num_classes=num_classes)

    hospital_loaders = []
    hospital_stats = []
    skipped_hospitals = []

    for i in range(num_hospitals):
        sub_indices = [train_train_indices[idx] for idx in client_train_splits[i]]
        hospital_name = fed_cfg["hospital_names"][i] if i < len(fed_cfg["hospital_names"]) else f"Hospital_{i}"

        if len(sub_indices) == 0:
            # Extreme Dirichlet skew (low alpha) combined with a small dataset can
            # legitimately assign zero samples to a client for a given random split.
            # Skip it rather than constructing an empty DataLoader (which crashes),
            # and make sure this is visible rather than silently dropped.
            skipped_hospitals.append(hospital_name)
            continue

        client_ds = Subset(MedicalDataset(images, labels, transform=train_transform), sub_indices)
        loader = DataLoader(client_ds, batch_size=batch_size, shuffle=True)

        # Calculate class distribution for hospital
        c_labels = [labels[idx] for idx in sub_indices]
        class_counts = {cls: c_labels.count(cls) for cls in range(num_classes)}

        hospital_loaders.append(loader)
        hospital_stats.append({
            "hospital_id": i,
            "hospital_name": hospital_name,
            "sample_count": len(sub_indices),
            "class_distribution": class_counts
        })

    if skipped_hospitals:
        print(
            f"[!] Warning: {len(skipped_hospitals)} hospital(s) received 0 training samples "
            f"under dirichlet_alpha={alpha} with only {num_train} training images available, "
            f"and were skipped: {', '.join(skipped_hospitals)}. This is expected extreme-Dirichlet "
            f"behavior on small datasets, not a bug - use more data (raise max_samples_per_class) "
            f"or a less extreme alpha if you need all hospitals represented."
        )
    if not hospital_loaders:
        raise ValueError(
            f"All {num_hospitals} hospitals received 0 training samples under dirichlet_alpha={alpha} "
            f"with only {num_train} training images available. Increase max_samples_per_class / use "
            f"more data, or use a higher (less extreme) dirichlet_alpha."
        )

    if len(test_indices) == 0:
        raise ValueError(
            f"The 80/20 train/test split left 0 test samples (only {num_samples} total images loaded). "
            f"Increase max_samples_per_class so there's enough data for a non-empty test set."
        )

    test_ds = Subset(MedicalDataset(images, labels, transform=test_transform), test_indices)
    global_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if return_val:
        if len(val_indices) > 0:
            val_ds = Subset(MedicalDataset(images, labels, transform=test_transform), val_indices)
            global_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            global_val_loader = None
        return hospital_loaders, global_val_loader, global_test_loader, hospital_stats
    else:
        return hospital_loaders, global_test_loader, hospital_stats
