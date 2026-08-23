import os

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _collect_images(folder: str):
    """Non-recursive collection of supported image files directly inside `folder`."""
    found = []
    for fname in sorted(os.listdir(folder)):
        full = os.path.join(folder, fname)
        if os.path.isfile(full) and fname.lower().endswith(SUPPORTED_EXTS):
            found.append(full)
    return found


def load_real_medical_dataset(data_dir: str, classes: list, max_samples_per_class: int = None):
    """
    Scans a folder-per-class directory structure and returns image file paths + integer labels.

    Supports two common layouts (class folder names must match `classes` from the config,
    case/space/underscore-insensitive):

        # Flat layout
        data_dir/
            Normal/
                img001.png
                ...

        # Kaggle "COVID-19 Radiography Database" layout (images nested one level
        # deeper under an `images/` subfolder, with a sibling `masks/` folder ignored)
        data_dir/
            Normal/
                images/
                    img001.png
                    ...
                masks/
                    ...

    Args:
        max_samples_per_class: if set, caps how many images are loaded per class
            (first N alphabetically). Useful for a quick full-pipeline sanity run
            on real data before committing to the full dataset - does NOT affect
            model architecture/checkpoint compatibility, only how much data is used.

    Returns:
        image_paths (list[str]), labels (list[int])
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Real dataset directory not found: {data_dir}\n"
            f"Expected one subfolder per class: {classes}"
        )

    # Map normalized folder name -> class index, so "covid-19", "COVID_19", "Viral Pneumonia" etc. still match
    def norm(s):
        return s.lower().replace(" ", "").replace("_", "").replace("-", "")

    class_lookup = {norm(c): i for i, c in enumerate(classes)}

    image_paths = []
    labels = []
    found_dirs = []

    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if not os.path.isdir(full_path):
            continue
        key = norm(entry)
        if key not in class_lookup:
            continue
        found_dirs.append(entry)
        label = class_lookup[key]

        # Prefer a nested "images" subfolder if present (Kaggle-style layout);
        # otherwise fall back to images directly inside the class folder.
        images_subdir = os.path.join(full_path, "images")
        if os.path.isdir(images_subdir):
            class_images = _collect_images(images_subdir)
        else:
            class_images = _collect_images(full_path)

        if max_samples_per_class is not None:
            class_images = class_images[:max_samples_per_class]

        for path in class_images:
            image_paths.append(path)
            labels.append(label)

    missing = [c for c in classes if norm(c) not in [norm(d) for d in found_dirs]]
    if missing:
        raise FileNotFoundError(
            f"Could not find a subfolder for class(es) {missing} inside {data_dir}. "
            f"Found subfolders: {found_dirs or 'none'}. "
            f"Each class in the config's dataset.classes list needs its own folder of images "
            f"(directly inside the class folder, or inside a nested 'images' subfolder)."
        )

    if len(image_paths) == 0:
        raise ValueError(
            f"No supported images ({', '.join(SUPPORTED_EXTS)}) found under {data_dir}."
        )

    return image_paths, labels

