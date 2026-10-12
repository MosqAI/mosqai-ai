import os
import shutil
import logging
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

from src.inference_engine import MosqAInferenceEngine
import cloudinary
import cloudinary.uploader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mosqai_ai")

app = FastAPI(
    title="MOSQAI AI Engine",
    description="MVP API",
    version="1.0.0"
)

# Cloudinary Setup
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET and CLOUDINARY_CLOUD_NAME != "your_cloud_name":
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "mosqai_local"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "your_actual_password"),
    "host": os.getenv("DB_HOST", "host.docker.internal"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

# In-memory tracking fallback when database is unavailable
IN_MEMORY_TRAP_LOG: dict[str, int] = {}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS, connect_timeout=2)
        return conn
    except Exception as e:
        logger.warning(f"Database connection not available ({e}); using in-memory state tracking.")
        return None

def ensure_db_table(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trap_analytics_log (
                    id SERIAL PRIMARY KEY,
                    trap_id VARCHAR(64) NOT NULL,
                    image_uid VARCHAR(255),
                    new_mosquitoes_count INT NOT NULL DEFAULT 0,
                    total_mosquitoes_count INT NOT NULL DEFAULT 0,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to auto-create table trap_analytics_log: {e}")
        try:
            conn.rollback()
        except Exception:
            pass

def upload_frame_asset(file_path: str, trap_id: str) -> str:
    if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET and CLOUDINARY_CLOUD_NAME != "your_cloud_name":
        try:
            upload_meta = cloudinary.uploader.upload(
                file_path,
                folder=f"mosqai_traps/trap_{trap_id}",
                overwrite=True
            )
            return upload_meta.get("public_id")
        except Exception as e:
            logger.warning(f"Cloudinary upload failed ({e}); falling back to local reference UID.")
    return f"local_ref_{trap_id}_{uuid.uuid4().hex[:8]}"

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
os.makedirs(TEMP_DIR, exist_ok=True)

class TrapResponseSchema(BaseModel):
    image_uid: Optional[str] = None
    new_mosquitoes: int

@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
async def device_ping_handshake():
    return {"status": "responsive", "engine": "ONNX_YOLOv8_Active"}

@app.post("/trap/frame", response_model=TrapResponseSchema, status_code=status.HTTP_201_CREATED)
@app.post("/api/v1/trap/frame", response_model=TrapResponseSchema, status_code=status.HTTP_201_CREATED)
async def process_incoming_trap_frame(
    trap_id: str = Form(...), 
    file: UploadFile = File(...)
):
    if AI_ENGINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Engine is not initialized. Model file missing."
        )

    local_file_path = os.path.join(TEMP_DIR, f"{trap_id}_incoming.jpg")
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_connection = None
    try:
        db_connection = get_db_connection()
        last_total_count = 0

        if db_connection:
            try:
                ensure_db_table(db_connection)
                with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT total_mosquitoes_count 
                        FROM trap_analytics_log 
                        WHERE trap_id = %s 
                        ORDER BY logged_at DESC LIMIT 1
                        """,
                        (trap_id,)
                    )
                    db_record = cursor.fetchone()
                    if db_record:
                        last_total_count = db_record["total_mosquitoes_count"]
            except Exception as e:
                logger.warning(f"Failed to query DB: {e}. Falling back to in-memory count.")
                try:
                    db_connection.rollback()
                except Exception:
                    pass
                last_total_count = IN_MEMORY_TRAP_LOG.get(trap_id, 0)
        else:
            last_total_count = IN_MEMORY_TRAP_LOG.get(trap_id, 0)
        
        current_total_count = AI_ENGINE.count_mosquitoes(local_file_path)
        
        if current_total_count <= last_total_count:
            return TrapResponseSchema(image_uid=None, new_mosquitoes=0)
            
        new_mosquitoes_delta = current_total_count - last_total_count
        image_uid = upload_frame_asset(local_file_path, trap_id)

        if db_connection:
            try:
                with db_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trap_analytics_log 
                        (trap_id, image_uid, new_mosquitoes_count, total_mosquitoes_count) 
                        VALUES (%s, %s, %s, %s)
                        """,
                        (trap_id, image_uid, new_mosquitoes_delta, current_total_count)
                    )
                db_connection.commit()
            except Exception as e:
                logger.warning(f"Failed to log to DB: {e}. Falling back to in-memory storage.")
                try:
                    db_connection.rollback()
                except Exception:
                    pass
                IN_MEMORY_TRAP_LOG[trap_id] = current_total_count
        else:
            IN_MEMORY_TRAP_LOG[trap_id] = current_total_count
        
        return TrapResponseSchema(image_uid=image_uid, new_mosquitoes=new_mosquitoes_delta)

    except Exception as server_error:
        if db_connection:
            try:
                db_connection.rollback()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution engine crash tracking: {str(server_error)}"
        )
    finally:
        if db_connection:
            try:
                db_connection.close()
            except Exception:
                pass
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

@app.post("/trap/reset", status_code=status.HTTP_200_OK)
@app.post("/api/v1/trap/reset", status_code=status.HTTP_200_OK)
async def trigger_trap_chamber_reset(trap_id: str = Form(...)):
    db_connection = None
    try:
        db_connection = get_db_connection()
        if db_connection:
            try:
                ensure_db_table(db_connection)
                with db_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trap_analytics_log 
                        (trap_id, image_uid, new_mosquitoes_count, total_mosquitoes_count) 
                        VALUES (%s, 'RESET_VECTOR', 0, 0)
                        """,
                        (trap_id,)
                    )
                db_connection.commit()
            except Exception as e:
                logger.warning(f"Failed to insert reset in DB: {e}. Resetting in-memory state.")
                try:
                    db_connection.rollback()
                except Exception:
                    pass
                IN_MEMORY_TRAP_LOG[trap_id] = 0
        else:
            IN_MEMORY_TRAP_LOG[trap_id] = 0

        return {"status": "reset_successful", "trap_id": trap_id, "tracking_total": 0}
        
    except Exception as sql_error:
        if db_connection:
            try:
                db_connection.rollback()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database state injection failure: {str(sql_error)}"
        )
    finally:
        if db_connection:
            try:
                db_connection.close()
            except Exception:
                pass
