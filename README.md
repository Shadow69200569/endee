<p align="center">
  <img src="C:\Users\ELCOT\.gemini\antigravity\brain\0db83f44-1073-492a-b9d6-3f161e8775eb\ui_screenshot.png" height="200" alt="Endee Vision UI">
</p>

<h1 align="center">🔍 Endee Vision — Large-Scale Image Similarity Search</h1>
<p align="center">
  A production-ready AI/ML system demonstrating real-world vector image search, powered by the <strong>Endee Vector Database</strong> and OpenAI's <strong>CLIP</strong> model.
</p>

---

## Project Overview

**Endee Vision** allow users to discover visually similar images from a massive dataset. It uses **OpenAI's CLIP (ViT-B/32)** model to convert images into 512-dimensional vectors and the **Endee Vector Database** for sub-millisecond similarity retrieval.

### Key Features
- 📊 **Scale:** Indexed **15,000+ images** across 10 categories.
- ⚡ **Speed:** High-performance vector retrieval using Endee's HNSW architecture.
- 🎨 **UX:** Modern Glassmorphism dark-mode UI with drag-and-drop support.
- 💾 **Persistence:** Includes a custom persistent mock server for Windows environments without Docker.

---

## Dataset: CIFAR-10 (High Volume)
To demonstrate large-scale capabilities, the system uses the **CIFAR-10** dataset:
- **15,020 Images** total.
- **10 Categories:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Preprocessing:** All images upscaled to 224x224 for optimal feature extraction by CLIP.

---

## How Endee Is Used

Endee serves as the core vector hosting layer. The system interacts with Endee via its **HTTP REST API**:

| Component | Endpoint | Use Case |
|---|---|---|
| **Index Creation** | `POST /api/v1/index/create` | Sets up a 512-dim cosine similarity index. |
| **Vector Insertion** | `POST /api/v1/index/{name}/vector/insert` | Bulk inserts CLIP embeddings with metadata. |
| **Similarity Search** | `POST /api/v1/index/{name}/search` | Retrieves top-K matches using HNSW logic. |
| **ID Management** | `GET /api/v1/index/{name}/vectors/ids` | Used for smart indexing/resume support. |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed.
- (Optional) Docker for running the real Endee C++ server.

### 2. Setup Database
If you don't have Docker, use our persistent mock server:
```bash
python scripts/mock_endee_server_persistent.py
```
*Note: This server saves the index to `dataset/endee_mock_db.json` so you never lose your data.*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Load Dataset & Index (Automatic)
```bash
# Download 15,000 images
python scripts/download_large_dataset.py

# Batch index into Endee (uses CLIP + model batching)
python embeddings/generate_embeddings_batch.py
```

### 5. Start the Web Application
```bash
python api/app.py
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## Project Architecture

```
[ Frontend: HTML/CSS/JS ] <-> [ Backend: FastAPI ] <-> [ Endee Vector DB (Port 8080) ]
                                      |
                               [ OpenAI CLIP Model ]
```

---

## API Documentation

- `GET /health` : Check status of API and Endee DB connection.
- `POST /upload` : Upload an image to find semantically similar matches.
- `GET /` : Serves the interactive web interface.

---

## Technology Stack
- **Database:** Endee Vector Database
- **Model:** OpenAI CLIP (ViT-B/32)
- **Backend:** FastAPI, Python, Uvicorn
- **Frontend:** Vanilla JS / CSS (Glassmorphism)
- **Library:** Torch, Torchvision, Msgpack, Pillow

---

## License
Licensed under Apache 2.0. Built for Endee Labs.
