import os
import requests

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")

def download_images():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Static Unsplash images organized by category for a good representation of classes
    images = [
        # Nature
        ("nature_01.jpg", "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&q=80"),
        ("nature_02.jpg", "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&q=80"),
        ("nature_03.jpg", "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&q=80"),
        ("nature_04.jpg", "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=800&q=80"),
        ("nature_05.jpg", "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=800&q=80"),
        
        # Architecture
        ("architecture_01.jpg", "https://images.unsplash.com/photo-1481026469463-66327c86e544?w=800&q=80"),
        ("architecture_02.jpg", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80"),
        ("architecture_03.jpg", "https://images.unsplash.com/photo-1499092346589-b9b6be3e94b2?w=800&q=80"),
        ("architecture_04.jpg", "https://images.unsplash.com/photo-1429497419816-9ca5cfb4571a?w=800&q=80"),
        ("architecture_05.jpg", "https://images.unsplash.com/photo-1460472178825-e5240623afd5?w=800&q=80"),
        
        # Cars
        ("cars_01.jpg", "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=800&q=80"),
        ("cars_02.jpg", "https://images.unsplash.com/photo-1556189250-72ba954cfc2b?w=800&q=80"),
        ("cars_03.jpg", "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=800&q=80"),
        ("cars_04.jpg", "https://images.unsplash.com/photo-1553440569-bcc63803a83d?w=800&q=80"),
        ("cars_05.jpg", "https://images.unsplash.com/photo-1525609004556-c46c7d6cf023?w=800&q=80"),
        
        # People
        ("people_01.jpg", "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80"),
        ("people_02.jpg", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80"),
        ("people_03.jpg", "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=800&q=80"),
        ("people_04.jpg", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80"),
        ("people_05.jpg", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=800&q=80")
    ]

    print(f"Downloading {len(images)} high-quality images from Unsplash...")
    for filename, url in images:
        path = os.path.join(DATASET_DIR, filename)
        if os.path.exists(path):
            print(f"Skipping {filename}, already exists.")
            continue
            
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    print(f"Dataset preparation complete! Files are stored in {DATASET_DIR}")

if __name__ == "__main__":
    download_images()
