# 🔍 Endee Vision — Large-Scale Image Similarity Search

A production-ready AI/ML system demonstrating real-world vector image search, powered by the **Endee Vector Database** and OpenAI's **CLIP** model.

![Endee Vision UI](./docs/images/ui_screenshot.webp)

---

## Project Overview

**Endee Vision** allows users to discover visually similar images from a massive dataset. It uses **OpenAI's CLIP (ViT-B/32)** model to convert images into 512-dimensional vectors and the **Endee Vector Database** for sub-millisecond similarity retrieval.

### Key Features
- 📊 **Scale:** Indexed **15,000+ images** across 10 categories.
- ⚡ **Speed:** High-performance vector retrieval using Endee's HNSW architecture.
- 🎨 **UX:** Modern Glassmorphism dark-mode UI with drag-and-drop support.
- 🤖 **Agentic Support:** Optimized for execution via the **Antigravity AI Agent**.

---

## Getting Started

Follow these steps to get the system running from scratch.

### 1. Clone the Repository
```bash
git clone https://github.com/Shadow69200569/endee.git
cd endee
```

### 2. Environment Setup
Create a virtual environment and install the necessary dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup the Database (Endee OSS)
The project requires the **Endee Vector Database** running at `localhost:8080`.

#### Option A: Docker (Recommended for Production)
```bash
docker-compose up -d
```
*This starts the high-performance C++ Endee engine via Docker.*

#### Option B: Persistent Mock Server (For Development)
If you cannot use Docker, run the provided mock server:
```bash
python scripts/mock_endee_server_persistent.py
```

### 4. Download and Prepare Dataset
For accurate search results, you must download the full 15,000 image dataset (CIFAR-10 based):
```bash
# Downloads and prepares ~15,000 images across 10 categories
python scripts/download_large_dataset.py
```
**Categories Included:** 
1. Airplane 
2. Automobile 
3. Bird 
4. Cat 
5. Deer 
6. Dog 
7. Frog 
8. Horse 
9. Ship 
10. Truck

### 5. Generate Vector Embeddings
Once images are downloaded, generate the CLIP embeddings and index them into Endee:
```bash
python embeddings/generate_embeddings_batch.py
```
*Note: If the process is interrupted, you can resume by adding `--skip N` where N is the number of already indexed items.*

### 6. Run the Web Application
```bash
python api/app.py
```
Visit **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🚀 Easy Execution with Antigravity

If you are using the **Antigravity AI Agent**, you can automate the entire setup and maintenance!

- **To Setup:** *"Hey Antigravity, setup the Endee Vision project from scratch."*
- **To Debug:** *"Antigravity, check if the indexing script is running and fix any connection issues."*
- **To Verify:** *"Antigravity, upload airplane_00001.jpg and show me a screenshot of the search results."*

Antigravity will handle terminal commands, environment variables, and browser verification for you.

---

## Project Architecture

```
[ Frontend: HTML/CSS/JS ] <-> [ Backend: FastAPI ] <-> [ Endee Vector DB (Port 8080) ]
                                       |
                                [ OpenAI CLIP Model ]
```

---

## Technology Stack
- **Database:** Endee Vector Database
- **Model:** OpenAI CLIP (ViT-B/32)
- **Backend:** FastAPI, Python, Uvicorn
- **Frontend:** Vanilla JS / CSS (Glassmorphism)
- **Infrastructure:** Docker, Docker-Compose

---

## License
Licensed under Apache 2.0. Built for Endee Labs.
