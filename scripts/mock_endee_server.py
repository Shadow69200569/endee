"""
Mock Endee Server — Pure Python implementation of the Endee HTTP API.

This server runs on port 8080 and implements:
  GET  /api/v1/health
  GET  /api/v1/index/list
  POST /api/v1/index/create
  POST /api/v1/index/{name}/vector/insert
  POST /api/v1/index/{name}/search

Vectors are stored in memory and searched using cosine similarity.
This replaces the real Endee C++ server for local / Docker-free development.
"""

import json
import math
import msgpack
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# ---- In-memory "database" ----
indexes: dict = {}  # { "admin/index_name": { "dim": int, "records": [...] } }

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EndeeHandler(BaseHTTPRequestHandler):
    # Suppress default request logging (we have our own)
    def log_message(self, format, *args):
        pass

    def _send_json(self, code: int, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_msgpack(self, code: int, data):
        body = msgpack.packb(data, use_bin_type=True)
        self.send_response(code)
        self.send_header("Content-Type", "application/msgpack")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _index_id(self, name: str) -> str:
        return f"admin/{name}"

    # ---- GET ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/health":
            print("[Endee Mock] GET /health")
            self._send_json(200, {"status": "ok"})

        elif path == "/api/v1/index/list":
            result = [{"name": k.split("/", 1)[1], "dimension": v["dim"]}
                      for k, v in indexes.items()]
            self._send_json(200, {"indexes": result})

        else:
            self._send_json(404, {"error": "Not found"})

    # ---- POST ----
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body_bytes = self._read_body()

        # --- Create index ---
        if path == "/api/v1/index/create":
            try:
                body = json.loads(body_bytes)
                name = body["index_name"]
                dim = body["dim"]
                iid = self._index_id(name)
                if iid in indexes:
                    self._send_json(409, {"error": "Index already exists"})
                    return
                indexes[iid] = {"dim": dim, "records": []}
                print(f"[Endee Mock] Created index '{name}' dim={dim}")
                self._send_json(200, "Index created successfully")
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # --- Insert vectors ---
        insert_prefix = "/api/v1/index/"
        insert_suffix = "/vector/insert"
        if path.startswith(insert_prefix) and path.endswith(insert_suffix):
            index_name = path[len(insert_prefix):-len(insert_suffix)]
            iid = self._index_id(index_name)
            try:
                payload = json.loads(body_bytes)
                records = payload if isinstance(payload, list) else [payload]
                if iid not in indexes:
                    self._send_json(404, {"error": "Index not found"})
                    return
                for rec in records:
                    # Parse meta string if needed
                    meta = rec.get("meta", "{}")
                    if isinstance(meta, str):
                        try: meta = json.loads(meta)
                        except: pass
                    indexes[iid]["records"].append({
                        "id": str(rec.get("id", "")),
                        "vector": rec.get("vector", []),
                        "meta": meta
                    })
                count = len(indexes[iid]["records"])
                print(f"[Endee Mock] Inserted {len(records)} vectors into '{index_name}' (total={count})")
                self.send_response(200)
                self.end_headers()
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # --- Search ---
        search_suffix = "/search"
        if path.startswith(insert_prefix) and path.endswith(search_suffix) and "/vector/" not in path:
            index_name = path[len(insert_prefix):-len(search_suffix)]
            iid = self._index_id(index_name)
            try:
                body = json.loads(body_bytes)
                query = body.get("vector", [])
                k = int(body.get("k", 5))

                if iid not in indexes:
                    self._send_json(404, {"error": "Index not found"})
                    return

                records = indexes[iid]["records"]
                if not records:
                    self._send_msgpack(200, [])
                    return

                # Compute cosine similarity for all records
                scored = []
                for rec in records:
                    sim = cosine_similarity(query, rec["vector"])
                    # distance = 1 - similarity   (0 = identical, 2 = opposite)
                    distance = 1.0 - sim
                    scored.append((distance, rec))

                # Sort ascending by distance (lower = more similar)
                scored.sort(key=lambda x: x[0])
                top_k = scored[:k]

                results = []
                for dist, rec in top_k:
                    meta_val = rec.get("meta", {})
                    if isinstance(meta_val, dict):
                        meta_str = json.dumps(meta_val)
                    else:
                        meta_str = str(meta_val)
                    results.append({
                        "id": rec["id"],
                        "dist": round(dist, 6),
                        "meta": meta_str
                    })

                print(f"[Endee Mock] Search '{index_name}' k={k} → {len(results)} results")
                self._send_msgpack(200, results)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "Not found"})


def run_mock_server(host="0.0.0.0", port=8080):
    server = HTTPServer((host, port), EndeeHandler)
    print(f"[Endee Mock] Server running at http://{host}:{port}")
    print("[Endee Mock] Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run_mock_server()
