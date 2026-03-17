<p align="center">
  <img src="docs/assets/logo-dark.svg" height="80" alt="Endee Logo">
</p>

<h1 align="center">🔍 Endee Vision — Image Similarity Search</h1>
<p align="center">
  A production-ready AI/ML system demonstrating real-world vector image search, powered by the <strong>Endee Vector Database</strong> and OpenAI's <strong>CLIP</strong> model.
</p>

---

## Project Overview

**Endee Vision** allows users to upload any image and instantly retrieve the most visually similar images stored in the database. It showcases a complete end-to-end AI pipeline — from image encoding with a state-of-the-art neural network to high-speed nearest-neighbor retrieval using the Endee Vector Database.

This project was built as an AI/ML internship demonstration for [Endee Labs](https://endee.io/).

---

## System Architecture

```
User Uploads Image
        │
        ▼
 CLIP (ViT-B/32) Encoder
 [OpenAI CLIP Model]
        │  512-dim floating-point vector
        ▼
Endee Vector Database
 [HTTP REST API on localhost:8080]
 [Cosine similarity HNSW index]
        │  Top-K nearest neighbors
        ▼
  FastAPI Backend
  [/upload endpoint]
        │  JSON response with image paths & similarity scores
        ▼
  Browser Frontend
  [Drag-and-drop UI, results grid]
```

---

## How Endee Is Used

Endee is used as the vector database. This project interacts with Endee entirely through its **HTTP REST API** (`database/endee_client.py`):

| Operation | Endee API Endpoint | Description |
|---|---|---|
| Create index | `POST /api/v1/index/create` | Creates a cosine-similarity index of 512 dimensions |
| Insert vectors | `POST /api/v1/index/{name}/vector/insert` | Bulk-inserts CLIP embeddings with image metadata |
| Search | `POST /api/v1/index/{name}/search` | Finds top-K nearest neighbors via HNSW |
| Health check | `GET /api/v1/health` | Verifies the database is online |

Endee stores vectors using **HNSW (Hierarchical Navigable Small World)** graphs with INT16 quantization and AVX2 SIMD acceleration, providing sub-millisecond search performance.

---

## Dataset

The demo dataset consists of **20 high-quality Unsplash images** across four categories:
- 🌿 **Nature** – forests, landscapes
- 🏛️ **Architecture** – buildings, cityscapes
- 🚗 **Cars** – sports and everyday vehicles
- 👤 **People** – portrait photography

Images are downloaded automatically via the setup script.

---

## Installation & Setup

### Prerequisites

1. **Python 3.9+**
2. **Endee Server** running at `http://localhost:8080`
   - **Windows** (this system): Use Docker
   ```bash
   docker run --ulimit nofile=100000:100000 -p 8080:8080 -v ./endee-data:/data --name endee-server endeeio/endee-server:latest
   ```
   - **Linux/macOS**: Use the install script:
   ```bash
   chmod +x ./install.sh ./run.sh
   ./install.sh --release --avx2
   ./run.sh
   ```
   See [docs/getting-started.md](./docs/getting-started.md) for full instructions.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Shadow69200569/endee.git
cd endee

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the image dataset
python scripts/download_dataset.py

# 4. Generate CLIP embeddings and insert into Endee
#    (Ensure Endee server is running first!)
python embeddings/generate_embeddings.py

# 5. Start the API and frontend
python api/app.py
```

Then open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## Running the System

```
                     ┌──────────────────────────┐
                     │  Step 1: Start Endee DB   │
                     │  (port 8080)              │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Step 2: Download dataset  │
                     │  python scripts/download  │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Step 3: Generate & index │
                     │  python embeddings/gen... │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Step 4: Start API server │
                     │  python api/app.py        │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Step 5: Open browser     │
                     │  http://localhost:8000    │
                     └──────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check for the API and Endee DB status |
| `POST` | `/upload` | Upload an image, returns top-K similar images |
| `GET`  | `/` | Serves the web frontend |

### Example

```bash
# Check health
curl http://localhost:8000/health

# Search with an image
curl -X POST http://localhost:8000/upload -F "file=@/path/to/my_image.jpg"
```

### Sample Response

```json
{
  "results": [
    { "id": "nature_01.jpg", "distance": 0.0342, "path": "/dataset/images/nature_01.jpg" },
    { "id": "nature_03.jpg", "distance": 0.1201, "path": "/dataset/images/nature_03.jpg" },
    { "id": "nature_05.jpg", "distance": 0.1888, "path": "/dataset/images/nature_05.jpg" }
  ]
}
```

---

## Project Structure

```
endee/
├── dataset/
│   └── images/              # Downloaded images
├── embeddings/
│   └── generate_embeddings.py # CLIP encoding + Endee indexing
├── database/
│   └── endee_client.py       # REST client for Endee API
├── search/
│   └── search_similar.py     # Search logic module
├── api/
│   └── app.py                # FastAPI application server
├── frontend/
│   ├── index.html            # UI shell
│   ├── style.css             # Glassmorphism dark UI styling
│   └── script.js             # Drag-drop, API calls, rendering
├── scripts/
│   └── download_dataset.py   # Image dataset downloader
├── docs/                     # Endee documentation
├── src/                      # Endee C++ source code
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Vector Database | **Endee** (HNSW + SIMD, HTTP API) |
| Embedding Model | **OpenAI CLIP** (ViT-B/32, 512-dim) |
| Backend | **FastAPI** + Uvicorn |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Language | Python 3.9+ |

---

## License

This project extends the Endee open-source database, licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for full terms.
