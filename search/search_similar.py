import sys
import os
import torch
import clip
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.endee_client import EndeeClient

INDEX_NAME = "image_features"

class ImageSearcher:
    def __init__(self):
        print("Initializing ImageSearcher...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.client = EndeeClient()
        
    def search_similar(self, image_path: str, top_k: int = 5):
        """
        Process a query image and return top K similar images from Endee.
        """
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image)
                features /= features.norm(dim=-1, keepdim=True)
            
            vector_list = features.cpu().numpy()[0].tolist()
            
            # Query Endee
            results = self.client.search(INDEX_NAME, query_vector=vector_list, k=top_k)
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
            
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_similar.py <image_path>")
        sys.exit(1)
        
    searcher = ImageSearcher()
    results = searcher.search_similar(sys.argv[1])
    
    print("\nSearch Results:")
    for i, res in enumerate(results):
        print(f"{i+1}. ID: {res.get('id')}, Distance: {res.get('dist')}")
