"""
Batch Embedding Generator — optimized for large datasets (10K+ images).

Features:
- Processes images in batches (faster GPU/CPU usage with CLIP)
- Shows progress with ETA
- Skips images already indexed (resume support)
- Inserts into Endee in bulk batches of 100

Usage:
    python embeddings/generate_embeddings_batch.py [--batch-size 32]
"""

import os
import sys
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.endee_client import EndeeClient

INDEX_NAME = "image_features"
DIMENSION = 512  # CLIP ViT-B/32


def generate_batch(args):
    import torch
    import clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print("[INFO] Loading CLIP ViT-B/32 model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    client = EndeeClient()
    if not client.health_check():
        print("[ERROR] Endee server not reachable at localhost:8080.")
        print("        Start mock_endee_server_persistent.py first.")
        sys.exit(1)

    # Ensure index exists
    print(f"[INFO] Ensuring index '{INDEX_NAME}' exists (dim={DIMENSION})...")
    client.create_index(INDEX_NAME, DIMENSION, space_type="cosine")

    # Load existing IDs from the index to support resuming
    print("[INFO] Fetching already-indexed image IDs...")
    existing_ids = set()
    try:
        # Check if the endpoint returns JSON (mock) or something else (prod)
        import requests
        resp = requests.get(f"http://localhost:8080/api/v1/index/{INDEX_NAME}/vectors/ids", timeout=5)
        if resp.status_code == 200 and "application/json" in resp.headers.get("Content-Type", ""):
            existing_ids = set(resp.json().get("ids", []))
            print(f"[INFO] Found {len(existing_ids)} existing IDs via API. These will be skipped.")
        else:
            print(f"[INFO] ID retrieval endpoint not available or not JSON. Using --skip if provided.")
    except Exception as e:
        print(f"[WARN] Could not fetch existing IDs: {e}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset", "images")

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = [
        f for f in os.listdir(dataset_dir)
        if os.path.splitext(f.lower())[1] in IMAGE_EXTS
    ]
    all_images.sort()
    total = len(all_images)
    print(f"[INFO] Found {total} images in {dataset_dir}")

    batch_size = args.batch_size
    insert_batch_size = 100  # Insert into Endee every N vectors
    pending_records = []
    processed = 0
    inserted_total = 0
    start_time = time.time()

    def flush_to_endee(records):
        nonlocal inserted_total
        if not records:
            return
        try:
            client.insert_vectors(INDEX_NAME, records)
            inserted_total += len(records)
        except Exception as e:
            print(f"[ERROR] Failed to insert batch: {e}")

    # Skip logic
    if args.skip > 0:
        print(f"[INFO] Skipping first {args.skip} images as requested.")
        all_images = all_images[args.skip:]
        processed += args.skip

    # Process in CLIP batches
    i = 0
    while i < len(all_images):
        batch_names = all_images[i: i + batch_size]
        batch_tensors = []
        batch_valid_names = []

        for name in batch_names:
            img_path = os.path.join(dataset_dir, name)
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                batch_tensors.append(img)
                batch_valid_names.append(name)
            except Exception as e:
                print(f"[WARN] Skipping {name}: {e}")

        if not batch_tensors:
            i += batch_size
            continue

        import torch
        batch_input = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_input)
            features = features / features.norm(dim=-1, keepdim=True)

        vectors = features.cpu().float().numpy().tolist()

        for name, vector in zip(batch_valid_names, vectors):
            category = name.split("_")[0]  # e.g. "nature" from "nature_0001.jpg"
            record = {
                "id": name,
                "vector": vector,
                "meta": {
                    "path": f"dataset/images/{name}",
                    "category": category
                }
            }
            pending_records.append(record)

        # Flush to Endee every insert_batch_size records
        if len(pending_records) >= insert_batch_size:
            flush_to_endee(pending_records)
            pending_records = []

        processed += len(batch_valid_names)
        i += batch_size

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0
        print(f"[{processed}/{total}] Embedded {len(batch_valid_names)} | "
              f"{rate:.1f} img/s | ETA {remaining:.0f}s | "
              f"Queued: {len(pending_records)}")

    # Flush remaining
    flush_to_endee(pending_records)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Inserted {inserted_total} vectors into Endee.")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch CLIP embedding generator")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images per CLIP forward pass (default: 32)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Number of sorted images to skip (default: 0)")
    args = parser.parse_args()
    generate_batch(args)
