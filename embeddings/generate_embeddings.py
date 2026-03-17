import os
import torch
import clip
from PIL import Image
import json
import sys

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.endee_client import EndeeClient

# Setup dimensions and config
INDEX_NAME = "image_features"
DIMENSION = 512 # CLIP ViT-B/32 generates 512-dimensional embeddings

def get_embeddings(dataset_path: str):
    print("Loading CLIP model 'ViT-B/32'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Init client and create index
    client = EndeeClient()
    if not client.health_check():
        print("Warning: Endee server is not reachable at localhost:8080. Start the server first.")
        print("Continuing embedding generation logic, but insertion will fail unless mocked.")
    
    print(f"Ensuring index '{INDEX_NAME}' exists...")
    try:
        client.create_index(INDEX_NAME, DIMENSION, space_type="cos")
    except Exception as e:
        print(f"Could not create index: {e}")
    
    vectors_to_insert = []
    
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        print(f"Dataset path {dataset_path} does not exist. Run downoad_dataset.py first.")
        return
        
    images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the dataset directory.")
        return
        
    print(f"Found {len(images)} images. Generating embeddings...")
    for idx, image_name in enumerate(images):
        img_path = os.path.join(dataset_path, image_name)
        
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model.encode_image(image)
                # Normalize features for cosine similarity equivalence if needed
                features /= features.norm(dim=-1, keepdim=True)
                
            vector_list = features.cpu().numpy()[0].tolist()
            
            # Format record for Endee
            record = {
                "id": image_name,
                "vector": vector_list,
                # Store relative path so frontend can access it
                "meta": {"path": f"dataset/images/{image_name}", "category": image_name.split('_')[0]}
            }
            vectors_to_insert.append(record)
            print(f"[{idx+1}/{len(images)}] Processed {image_name}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            
    # Insert batch into Endee
    if vectors_to_insert:
        print(f"Inserting {len(vectors_to_insert)} vectors into Endee database...")
        try:
            client.insert_vectors(INDEX_NAME, vectors_to_insert)
            print("Successfully inserted all vectors.")
        except Exception as e:
            print(f"Failed to insert vectors: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset", "images")
    get_embeddings(dataset_dir)
