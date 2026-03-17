"""
Mock Endee Server with disk persistence.

Same HTTP API as before but saves/loads the index from JSON on disk,
so you don't need to re-run generate_embeddings.py after every restart.

Usage:
    python scripts/mock_endee_server_persistent.py
"""

import json
import math
import os
import msgpack
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "dataset", "endee_mock_db.json")

# ---- In-memory store with disk persistence ----
indexes: dict = {}


def load_from_disk():
    global indexes
    if os.path.exists(DATA_FILE):
        print(f"[Endee Mock] Loading index from {DATA_FILE}...")
        with open(DATA_FILE, "r") as f:
            indexes = json.load(f)
        for iid, v in indexes.items():
            total = len(v.get("records", []))
            print(f"[Endee Mock]   {iid}: {total} vectors")
        print("[Endee Mock] Index loaded.")
    else:
        print("[Endee Mock] No saved index found, starting fresh.")


def save_to_disk():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(indexes, f)


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EndeeHandler(BaseHTTPRequestHandler):
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

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/v1/health":
            total = sum(len(v.get("records", [])) for v in indexes.values())
            self._send_json(200, {"status": "ok", "total_vectors": total})
        elif path == "/api/v1/index/list":
            result = [{"name": k.split("/", 1)[1], "dimension": v["dim"],
                       "total_elements": len(v.get("records", []))}
                      for k, v in indexes.items()]
            self._send_json(200, {"indexes": result})
        
        elif path.startswith(prefix) and path.endswith("/vectors/ids"):
            name = path[len(prefix):-len("/vectors/ids")]
            iid = self._index_id(name)
            if iid in indexes:
                ids = [r["id"] for r in indexes[iid]["records"]]
                self._send_json(200, {"ids": ids})
            else:
                self._send_json(404, {"error": "Index not found"})
        
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        body_bytes = self._read_body()

        if path == "/api/v1/index/create":
            try:
                body = json.loads(body_bytes)
                iid = self._index_id(body["index_name"])
                if iid in indexes:
                    self._send_json(409, {"error": "Already exists"})
                    return
                indexes[iid] = {"dim": body["dim"], "records": []}
                save_to_disk()
                print(f"[Endee Mock] Index created: {body['index_name']}")
                self._send_json(200, "Index created successfully")
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        insert_suffix = "/vector/insert"
        search_suffix = "/search"
        prefix = "/api/v1/index/"

        if path.startswith(prefix) and path.endswith(insert_suffix):
            name = path[len(prefix):-len(insert_suffix)]
            iid = self._index_id(name)
            try:
                payload = json.loads(body_bytes)
                records = payload if isinstance(payload, list) else [payload]
                if iid not in indexes:
                    self._send_json(404, {"error": "Index not found"})
                    return
                for rec in records:
                    meta = rec.get("meta", "{}")
                    if isinstance(meta, str):
                        try: meta = json.loads(meta)
                        except: pass
                    indexes[iid]["records"].append({
                        "id": str(rec.get("id", "")),
                        "vector": rec.get("vector", []),
                        "meta": meta
                    })
                # Save after every batch insert
                save_to_disk()
                count = len(indexes[iid]["records"])
                print(f"[Endee Mock] Inserted {len(records)} → '{name}' total={count}")
                self.send_response(200)
                self.end_headers()
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        if path.startswith(prefix) and path.endswith(search_suffix) and "/vector/" not in path:
            name = path[len(prefix):-len(search_suffix)]
            iid = self._index_id(name)
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
                scored = [(1.0 - cosine_similarity(query, rec["vector"]), rec)
                          for rec in records]
                scored.sort(key=lambda x: x[0])
                results = []
                for dist, rec in scored[:k]:
                    meta_val = rec.get("meta", {})
                    meta_str = json.dumps(meta_val) if isinstance(meta_val, dict) else str(meta_val)
                    results.append({"id": rec["id"], "dist": round(dist, 6), "meta": meta_str})
                print(f"[Endee Mock] Search '{name}' k={k} → {len(results)} results")
                self._send_msgpack(200, results)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "Not found"})


def run(host="0.0.0.0", port=8080):
    load_from_disk()
    server = HTTPServer((host, port), EndeeHandler)
    total = sum(len(v.get("records", [])) for v in indexes.values())
    print(f"[Endee Mock] Server running at http://{host}:{port}  (vectors in memory: {total})")
    server.serve_forever()


if __name__ == "__main__":
    run()
