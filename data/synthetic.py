import numpy as np
from PIL import Image

def generate_synthetic_medical_xray(label: int, image_size: int = 224) -> Image.Image:
    """
    Generates unique realistic synthetic chest X-ray images for testing and validation.
    Label 0: Normal (just ribs + noise)
    Label 1: Lung_Opacity (focal opacity, a region with increased intensity)
    Label 2: Viral Pneumonia (diffuse opacities, multiple smaller regions of increased intensity)
    Label 3: COVID (bilateral peripheral ground-glass opacities, multiple regions on the left and right sides)
    """
    # Use current numpy global random state (reproducible if seeded outside)
    base = np.random.normal(loc=120, scale=25, size=(image_size, image_size)).astype(np.float32)
    
    # Add rib structures (with some random variations in angle/spacing)
    x = np.linspace(-3, 3, image_size)
    y = np.linspace(-3, 3, image_size)
    xx, yy = np.meshgrid(x, y)
    
    # Slight random variation in ribs to avoid exact identical rib lines
    freq_x = 4.0 + np.random.uniform(-0.2, 0.2)
    freq_y = 2.0 + np.random.uniform(-0.1, 0.1)
    ribs = np.sin(xx * freq_x) * np.cos(yy * freq_y) * 30
    base += ribs

    # Label-specific opacities with random position/intensity to prevent trivial overfitting
    if label == 1: # Lung_Opacity (focal opacity)
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        size = np.random.uniform(0.4, 0.7)
        intensity = np.random.uniform(40.0, 70.0)
        mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / size) * intensity
        base += mask
    elif label == 2: # Viral Pneumonia (diffuse opacities)
        # Generate 3-5 small random opacity spots
        num_spots = np.random.randint(3, 6)
        for _ in range(num_spots):
            cx = np.random.uniform(-1.5, 1.5)
            cy = np.random.uniform(-1.5, 1.5)
            size = np.random.uniform(0.15, 0.3)
            intensity = np.random.uniform(30.0, 50.0)
            mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / size) * intensity
            base += mask
    elif label == 3: # COVID (bilateral peripheral opacities)
        # Left side opacity
        cx1 = np.random.uniform(-1.5, -0.8)
        cy1 = np.random.uniform(0.0, 1.2)
        size1 = np.random.uniform(0.3, 0.6)
        intensity1 = np.random.uniform(45.0, 65.0)
        mask1 = np.exp(-((xx - cx1)**2 + (yy - cy1)**2) / size1) * intensity1
        
        # Right side opacity
        cx2 = np.random.uniform(0.8, 1.5)
        cy2 = np.random.uniform(0.0, 1.2)
        size2 = np.random.uniform(0.3, 0.6)
        intensity2 = np.random.uniform(45.0, 65.0)
        mask2 = np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / size2) * intensity2
        
        base += mask1 + mask2

    base = np.clip(base, 0, 255).astype(np.uint8)
    img_rgb = np.stack([base, base, base], axis=-1)
    return Image.fromarray(img_rgb)

