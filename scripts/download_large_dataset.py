"""
Large Dataset Downloader using STL-10 via torchvision.

STL-10 dataset provides:
  - 10 categories: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
  - 500 training images + 800 test images = 1,300 per class (13,000 total)
  - 96x96 pixels (high quality, suitable for CLIP)
  - Downloads automatically from AWS — no API key required

Usage:
    python scripts/download_large_dataset.py

The script saves images to:
    dataset/images/{category_name}_{index:04d}.jpg
"""

import os
import sys
from io import BytesIO
from PIL import Image

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
STL10_CACHE_DIR = os.path.join(BASE_DIR, "dataset", "stl10_cache")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(STL10_CACHE_DIR, exist_ok=True)

def download_stl10():
    """Download STL-10 via torchvision and save as individual JPEG files."""
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        import numpy as np
    except ImportError as e:
        print(f"Error: required package not found: {e}")
        print("Run: pip install torchvision pillow")
        sys.exit(1)

    # STL-10 class labels
    classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    print("="*60)
    print("Downloading STL-10 dataset via torchvision...")
    print("  10 categories × ~1300 images = ~13,000 images total")
    print("  This may take several minutes (2.6 GB download)")
    print("="*60)

    # Count existing images to allow resuming
    existing = {}
    for fname in os.listdir(DATASET_DIR):
        for cat in classes:
            if fname.startswith(cat + "_"):
                existing[cat] = existing.get(cat, 0) + 1

    counters = {cat: existing.get(cat, 0) for cat in classes}
    if any(v > 0 for v in counters.values()):
        print("Found existing images, will resume download.")
        for cat, cnt in counters.items():
            if cnt > 0:
                print(f"  {cat}: {cnt} already downloaded")

    # Download train split
    for split in ["train", "test"]:
        print(f"\nLoading {split} split...")
        try:
            dataset = datasets.STL10(
                root=STL10_CACHE_DIR,
                split=split,
                download=True,
                transform=None  # Raw PIL images
            )
        except Exception as e:
            print(f"Failed to download STL-10 {split}: {e}")
            continue

        total = len(dataset)
        print(f"  {total} images in {split} split. Extracting...")

        for i, (img, label) in enumerate(dataset):
            cat = classes[label]
            counters[cat] += 1
            idx = counters[cat]

            out_path = os.path.join(DATASET_DIR, f"{cat}_{idx:04d}.jpg")
            if os.path.exists(out_path):
                continue  # Skip if already saved

            # PIL image → save as JPEG at 224x224 (good for CLIP)
            img_resized = img.resize((224, 224), Image.LANCZOS)
            img_resized.save(out_path, "JPEG", quality=90)

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] Saved {out_path}")

    print("\n" + "="*60)
    print("Dataset extraction complete!")
    total_saved = sum(counters.values())
    print(f"Total images saved: {total_saved}")
    print(f"Output directory: {DATASET_DIR}")
    for cat, cnt in sorted(counters.items()):
        print(f"  {cat:12s}: {cnt} images")
    print("="*60)


if __name__ == "__main__":
    download_stl10()
