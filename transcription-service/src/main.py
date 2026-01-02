from fastapi import FastAPI,UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from src.asr_engine import CodeSwitchASR
import shutil
import os
import uuid

TEMP_DIR = "tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["SB_DISABLE_K2"] = "1"
asr_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_pipeline
    print("[Server] Initializing ASR Pipeline...")
    asr_pipeline = CodeSwitchASR()
    yield
    print("[Server] Shutting down ASR Pipeline...")

app = FastAPI(title="Telugu-English ASR API", lifespan=lifespan)
    
@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    if not asr_pipeline:
        raise HTTPException(status_code=500, detail="Models are still loading...")
    
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, unique_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[Server] Processing: {unique_name}")
        transcript = asr_pipeline.process_file(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "transcription": transcript
        }
        
    except Exception as e:
        print(f"[Server] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)