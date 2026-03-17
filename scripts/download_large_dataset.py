"""
Large Dataset Downloader using CIFAR-10 via torchvision.

CIFAR-10 dataset provides:
  - 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - 6,000 images per class (50,000 train + 10,000 test)
  - 32x32 pixels (upscaled to 224x224 for CLIP)
  - Downloads automatically (~163 MB) — MUCH faster than STL-10.

Usage:
    python scripts/download_large_dataset.py

The script saves images to:
    dataset/images/{category_name}_{index:05d}.jpg
"""

import os
import sys
from PIL import Image

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
CIFAR_CACHE_DIR = os.path.join(BASE_DIR, "dataset", "cifar_cache")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CIFAR_CACHE_DIR, exist_ok=True)

def download_cifar10():
    """Download CIFAR-10 via torchvision and save as JPEG files."""
    try:
        import torchvision.datasets as datasets
        import numpy as np
    except ImportError as e:
        print(f"Error: required package not found: {e}")
        print("Run: pip install torchvision pillow")
        sys.exit(1)

    # CIFAR-10 class labels
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("="*60)
    print("Downloading CIFAR-10 dataset via torchvision...")
    print("  10 categories × 6,000 images = 60,000 images total")
    print("  Size: ~163 MB (Much faster than STL-10)")
    print("="*60)

    counters = {cat: 0 for cat in classes}

    # Download both train and test splits
    for train_mode in [True, False]:
        split_name = "train" if train_mode else "test"
        print(f"\nLoading {split_name} split...")
        try:
            dataset = datasets.CIFAR10(
                root=CIFAR_CACHE_DIR,
                train=train_mode,
                download=True
            )
        except Exception as e:
            print(f"Failed to download CIFAR-10 {split_name}: {e}")
            continue

        total = len(dataset)
        print(f"  {total} images in {split_name} split. Extracting...")

        for i, (img, label) in enumerate(dataset):
            cat = classes[label]
            counters[cat] += 1
            idx = counters[cat]

            # Limit to ~1500 images per category as requested (1000-2000)
            # to keep indexing time reasonable (indexing 60k images in mock DB might be slow)
            if idx > 1500:
                continue

            out_path = os.path.join(DATASET_DIR, f"{cat}_{idx:05d}.jpg")
            if os.path.exists(out_path):
                continue

            # Upscale 32x32 to 224x224 for better CLIP feature extraction
            img_resized = img.resize((224, 224), Image.LANCZOS)
            img_resized.save(out_path, "JPEG", quality=85)

            if (i + 1) % 2000 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] Saving {cat} images... (Current {cat} count: {idx})")

    print("\n" + "="*60)
    print("Dataset extraction complete!")
    total_saved = sum(min(v, 1500) for v in counters.values())
    print(f"Total images saved: {total_saved}")
    print(f"Output directory: {DATASET_DIR}")
    for cat, cnt in sorted(counters.items()):
        print(f"  {cat:12s}: {min(cnt, 1500)} images")
    print("="*60)


if __name__ == "__main__":
    download_cifar10()
