import os
import requests
import msgpack
import json
from typing import List, Dict, Any

class EndeeClient:
    """Client wrapper for interacting with Endee Vector Database via its REST API."""
    def __init__(self, host: str = "http://localhost:8080"):
        self.base_url = f"{host}/api/v1"
        self.headers = {"Content-Type": "application/json"}

    def health_check(self) -> bool:
        """Check if Endee server is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.RequestException:
            return False

    def list_indexes(self) -> List[Dict]:
        """List all available indexes."""
        response = requests.get(f"{self.base_url}/index/list", headers=self.headers)
        response.raise_for_status()
        return response.json().get("indexes", [])

    def create_index(self, index_name: str, dimension: int, space_type: str = "cosine", ef_con: int = 200, m: int = 16) -> bool:
        """Create a new index for vectors."""
        payload = {
            "index_name": index_name,
            "dim": dimension,
            "space_type": space_type,
            "ef_con": ef_con,
            "M": m
        }
        response = requests.post(f"{self.base_url}/index/create", json=payload, headers=self.headers)
        
        # 409 means it already exists, which is fine
        if response.status_code == 409:
            print(f"Index '{index_name}' already exists.")
            return True
            
        response.raise_for_status()
        return True

    def delete_index(self, index_name: str) -> bool:
        """Delete an index."""
        response = requests.delete(f"{self.base_url}/index/{index_name}/delete", headers=self.headers)
        return response.status_code == 200

    def insert_vectors(self, index_name: str, records: List[Dict[str, Any]]) -> bool:
        """
        Insert a batch of vectors.
        Records should be a list of dicts:
        [ {"id": "img1", "vector": [0.1, 0.2...], "meta": '{"path": "dataset/images/img1.jpg"}'} ]
        """
        # Note: 'meta' must be a JSON string, so we ensure it is
        for record in records:
            if "meta" in record and isinstance(record["meta"], dict):
                record["meta"] = json.dumps(record["meta"])

        response = requests.post(
            f"{self.base_url}/index/{index_name}/vector/insert", 
            json=records, 
            headers=self.headers
        )
        response.raise_for_status()
        return True

    def search(self, index_name: str, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        Search for top K similar vectors. Returns results decoded from MessagePack.
        """
        payload = {
            "k": k,
            "vector": query_vector
        }
        response = requests.post(
            f"{self.base_url}/index/{index_name}/search",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        
        # The Endee Database returns search results as application/msgpack
        try:
            results = msgpack.unpackb(response.content, raw=False)
            return results
        except Exception as e:
            print(f"Failed to decode msgpack response: {e}")
            return []
