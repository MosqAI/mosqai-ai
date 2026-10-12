import os 
import shutil 
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from src.inference_engine import MosqAInferenceEngine

import cloudinary 
import cloudinary.uploader 

app = FastAPI(
    title="MOSQAI AI Engine",
    description="MVP API",
    version="1.0.0"
)

cloudinary.config( 
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "Put here the Cloud name"), 
  api_key = os.getenv("CLOUDINARY_API_KEY", "Cloudinary API key"), 
  api_secret = os.getenv("CLOUDINARY_API_SECRET", "Cloudinary API SECRET"),
  secure = True
)

db_params = {
    "db_name":"mosqai_local",
    "user":"postgres",
    "password":"password_here",
    "host":"localhost",
    "port":5432
}

MODEL_PATH = "models/yolov8n_mosquito.onnx"

if not os.path.exists(MODEL_PATH):
    print("\n" + "!" * 60)
    print(f"[CRITICAL CONFIG ERROR]: ONNX Model file not found at: {MODEL_PATH}")
    print("Please verify that the file exists and is located in the 'models' directory.")
    print("!" * 60 + "\n")
    AI_ENGINE = None

else:
    print(f"[BOOTSTRAP]: Loading ONNX model layers from {MODEL_PATH}...")
    AI_ENGINE = MosqAInferenceEngine(model_path=MODEL_PATH)

TEMP_DIR = "tmp_frames"
os.makedirs(TEMP_DIR,exist_ok=True)

class TrapResponseSchema(BaseModel):
    image_uid: Optional[str] = None
    new_mosquitoes: int

@app.get("/api/v1/health",status_code=status.HTTP_200_OK)
async def device_ping_handshake():
    return {"status": "responsive", "engine": "ONNX_YOLOv8_Active"}

@app.post("/api/v1/trap/frame",response_model= TrapResponseSchema,status_code=status.HTTP_201_CREATED)
async def process_incoming_trap_frame(
    trap_id: str = Form(...), 
    file: UploadFile = File(...)
):
    local_file_path = os.path.join(TEMP_DIR, f"{trap_id}_incoming.jpg")
    
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_connection = None
    try:
        db_connection = psycopg2.connect(**DB_PARAMS)
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            (trap_id,)
        )
        db_record = cursor.fetchone()
        last_total_count = db_record["total_mosquitoes_count"] if db_record else 0
        
        current_total_count = AI_ENGINE.count_mosquitoes(local_file_path)
        
        if current_total_count <= last_total_count:
            return TrapResponseSchema(image_uid=None, new_mosquitoes=0)
            
        new_mosquitoes_delta = current_total_count - last_total_count
        
        upload_meta = cloudinary.uploader.upload(
            local_file_path,
            folder=f"mosqai_traps/trap_{trap_id}",
            overwrite=True
        )
        cloudinary_uid = upload_meta.get("public_id")
        
        cursor.execute(
            (trap_id, cloudinary_uid, new_mosquitoes_delta, current_total_count)
        )
        db_connection.commit()
        cursor.close()
        
        return TrapResponseSchema(image_uid=cloudinary_uid, new_mosquitoes=new_mosquitoes_delta)

    except Exception as server_error:
        if db_connection:
            db_connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution engine crash tracking: {str(server_error)}"
        )
        
    finally:
        if db_connection:
            db_connection.close()
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


@app.post("/api/v1/trap/reset", status_code=status.HTTP_200_OK)
async def trigger_trap_chamber_reset(trap_id: str = Form(...)):
    db_connection = None
    try:
        db_connection = psycopg2.connect(**DB_PARAMS)
        cursor = db_connection.cursor()
        
        # Append an empty state log marker entry
        cursor.execute(
            (trap_id,)
        )
        db_connection.commit()
        cursor.close()
        return {"status": "reset_successful", "trap_id": trap_id, "tracking_total": 0}
        
    except Exception as sql_error:
        if db_connection:
            db_connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database state injection failure: {str(sql_error)}"
        )
    finally:
        if db_connection:
            db_connection.close()