import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json

app_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(app_dir)
sys.path.append(base_dir)

from search.search_similar import ImageSearcher, INDEX_NAME

app = FastAPI(title="Endee Image Similarity Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attempt to load searcher model once at startup (this can be slow)
searcher = None
@app.on_event("startup")
def startup_event():
    global searcher
    print("Loading CLIP model and initializing search engine...")
    try:
        searcher = ImageSearcher()
    except Exception as e:
        print(f"Failed to load ImageSearcher. Will retry on first request. Error: {e}")

# Static file serving
dataset_dir = os.path.join(base_dir, "dataset")
frontend_dir = os.path.join(base_dir, "frontend")

os.makedirs(frontend_dir, exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)

# Mount paths
app.mount("/dataset", StaticFiles(directory=dataset_dir), name="dataset")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/health")
def health_check():
    global searcher
    db_ok = False
    if searcher and searcher.client:
        db_ok = searcher.client.health_check()
    return {"status": "ok", "db_connected": db_ok, "model_loaded": searcher is not None}

@app.post("/upload")
async def upload_for_search(file: UploadFile = File(...)):
    global searcher
    if searcher is None:
        try:
            searcher = ImageSearcher()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to initialize search engine: {e}"})
            
    if file.filename == "":
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
        
    # Save temp file
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Searching similar images for {temp_path}")
        results = searcher.search_similar(temp_path, top_k=6) # Get top 6
        
        # Parse output from MessagePack results
        formatted_results = []
        for res in results:
            try:
                # Endee msgpack results for `include_vectors=False` (default)
                # Usually {"id": "str", "dist": 0.12, "meta": "{\"path\":\"...\"}"}
                # Wait, Endee msgpack returns object with fields maybe different.
                # Assuming id, dist, meta are present or it's a list.
                meta_str = res.get("meta", "{}")
                if isinstance(meta_str, str):
                    meta_dict = json.loads(meta_str)
                else: 
                    meta_dict = meta_str
                    
                path = meta_dict.get("path", "")
                
                formatted_results.append({
                    "id": res.get("id"),
                    "distance": round(float(res.get("dist", 0.0)), 4),
                    "path": f"/{path}" # Serve locally via mounted path
                })
            except Exception as parse_e:
                print(f"Error parsing result config: {res} -> {parse_e}")
        
        return {"results": formatted_results}
        
    except Exception as e:
        print(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Serve root html
from fastapi.responses import FileResponse
@app.get("/")
def serve_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
