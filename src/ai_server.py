import os
import logging
import secrets
import tempfile
import threading
from collections import defaultdict
from typing import Optional
from fastapi import Depends, FastAPI, UploadFile, File, Form, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
import requests
from dotenv import load_dotenv

from src.inference_engine import MosqAInferenceEngine
import cloudinary
import cloudinary.uploader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mosqai_ai")

app = FastAPI(
    title="MOSQAI AI Engine",
    description="MVP API",
    version="1.0.0"
)

# Trap ids end up in Cloudinary folders and database filters, so keep them plain.
TRAP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

# Shared secret the device (and backend) must send in the X-Service-Token header.
SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "")
if not SERVICE_TOKEN:
    logger.error("AI_SERVICE_TOKEN is not set; trap endpoints will refuse every request.")


def require_service_token(x_service_token: str = Header(default="")):
    if not SERVICE_TOKEN:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service token not configured.")
    if not secrets.compare_digest(x_service_token.encode(), SERVICE_TOKEN.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid service token.")


# Cloudinary Setup
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
CLOUDINARY_ENABLED = bool(CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET)

if CLOUDINARY_ENABLED:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
else:
    logger.warning("Cloudinary credentials not configured; no images will be stored.")

# Collect mode uploads every frame to the dataset folder (for labelling and
# training), not only the frames with new mosquitoes.
COLLECT_MODE = os.getenv("COLLECT_MODE", "false").strip().lower() in ("1", "true", "yes", "on")
if COLLECT_MODE:
    logger.info("Collect mode on: every frame is uploaded to mosqai_dataset/.")

# Supabase (PostgREST), accessed with the server-side secret key (see db/schema.sql).
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
TRAP_LOG_ENDPOINT = f"{SUPABASE_URL}/rest/v1/trap_analytics_log"
SUPABASE_TIMEOUT = 5
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)
if not SUPABASE_ENABLED:
    logger.warning("Supabase credentials not configured; trap history is kept in memory only.")

# A new total is only accepted once it is the median of this many recent frames,
# so a single false-positive frame cannot raise the baseline. 1 disables smoothing.
COUNT_SMOOTHING_WINDOW = max(1, int(os.getenv("COUNT_SMOOTHING_WINDOW", "3")))


class TrapHistoryUnavailable(Exception):
    pass


class SupabaseTrapStore:
    def headers(self) -> dict:
        return {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

    def recent_events(self, trap_id: str, limit: int) -> list[dict]:
        try:
            response = requests.get(
                TRAP_LOG_ENDPOINT,
                headers=self.headers(),
                params={
                    "select": "event_type,detected_count,total_mosquitoes_count",
                    "trap_id": f"eq.{trap_id}",
                    "order": "logged_at.desc,id.desc",
                    "limit": limit,
                },
                timeout=SUPABASE_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise TrapHistoryUnavailable(f"Supabase read failed: {e}") from e

    def insert(self, row: dict):
        try:
            response = requests.post(
                TRAP_LOG_ENDPOINT,
                headers={**self.headers(), "Prefer": "return=minimal"},
                json=row,
                timeout=SUPABASE_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise TrapHistoryUnavailable(f"Supabase write failed: {e}") from e


class InMemoryTrapStore:
    def __init__(self):
        self.events: dict[str, list[dict]] = defaultdict(list)

    def recent_events(self, trap_id: str, limit: int) -> list[dict]:
        return list(reversed(self.events[trap_id][-limit:]))

    def insert(self, row: dict):
        history = self.events[row["trap_id"]]
        history.append(row)
        del history[:-COUNT_SMOOTHING_WINDOW]


TRAP_STORE = SupabaseTrapStore() if SUPABASE_ENABLED else InMemoryTrapStore()

# Serialises read-compute-write per trap within this process (run a single worker).
TRAP_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def upload_frame_asset(file_path: str, folder: str, context: Optional[dict] = None) -> Optional[str]:
    """Upload to Cloudinary and return the public_id, or None if it is not stored."""
    if not CLOUDINARY_ENABLED:
        return None
    try:
        upload_meta = cloudinary.uploader.upload(file_path, folder=folder, context=context)
        return upload_meta.get("public_id")
    except Exception as e:
        logger.error(f"Cloudinary upload to {folder} failed: {e}")
        return None


def compute_new_total(detected_count: int, recent_events: list[dict]) -> tuple[int, int]:
    """Return (new_mosquitoes, total) for a frame given the trap's newest-first history."""
    baseline = recent_events[0]["total_mosquitoes_count"] if recent_events else 0

    window = [detected_count]
    for event in recent_events:
        if event["event_type"] == "reset" or len(window) >= COUNT_SMOOTHING_WINDOW:
            break
        window.append(event["detected_count"])
    # Frames before a reset or a new trap's first frame saw an empty chamber, so an
    # increase must show up in a majority of the window before it is counted.
    window += [0] * (COUNT_SMOOTHING_WINDOW - len(window))
    smoothed = sorted(window)[(len(window) - 1) // 2]

    if smoothed <= baseline:
        return 0, baseline
    return smoothed - baseline, smoothed


MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n_mosquito.onnx")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
TARGET_CLASS_IDS = [int(c) for c in os.getenv("TARGET_CLASS_IDS", "").split(",") if c.strip()] or None

if not os.path.exists(MODEL_PATH):
    logger.critical(f"ONNX model file not found at {MODEL_PATH}; frame endpoint will return 503.")
    AI_ENGINE = None
else:
    logger.info(f"Loading ONNX model {MODEL_VERSION} from {MODEL_PATH}...")
    AI_ENGINE = MosqAInferenceEngine(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        class_ids=TARGET_CLASS_IDS,
    )

TEMP_DIR = "tmp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

class TrapResponseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_uid: Optional[str] = None
    new_mosquitoes: int
    model_version: str
    # Set only in collect mode: where this frame was stored for the training dataset.
    dataset_uid: Optional[str] = None

@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
def device_ping_handshake():
    return {
        "status": "responsive",
        "engine": "ONNX_YOLOv8_Active" if AI_ENGINE else "model_missing",
        "model_version": MODEL_VERSION,
    }

# Handlers are plain `def` so FastAPI runs the blocking inference and HTTP calls in
# its threadpool instead of stalling the event loop.
@app.post("/trap/frame", response_model=TrapResponseSchema, status_code=status.HTTP_201_CREATED,
          dependencies=[Depends(require_service_token)])
@app.post("/api/v1/trap/frame", response_model=TrapResponseSchema, status_code=status.HTTP_201_CREATED,
          dependencies=[Depends(require_service_token)])
def process_incoming_trap_frame(
    trap_id: str = Form(..., pattern=TRAP_ID_PATTERN),
    file: UploadFile = File(...)
):
    if AI_ENGINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Engine is not initialized. Model file missing."
        )

    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large.")

    with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".jpg", delete=False) as buffer:
        buffer.write(payload)
        local_file_path = buffer.name

    try:
        with TRAP_LOCKS[trap_id]:
            recent_events = TRAP_STORE.recent_events(trap_id, COUNT_SMOOTHING_WINDOW)
            detected_count = AI_ENGINE.count_mosquitoes(local_file_path)
            new_mosquitoes, total = compute_new_total(detected_count, recent_events)

            image_uid = None
            if new_mosquitoes:
                image_uid = upload_frame_asset(local_file_path, f"mosqai_traps/trap_{trap_id}")

            dataset_uid = None
            if COLLECT_MODE:
                dataset_uid = upload_frame_asset(
                    local_file_path,
                    f"mosqai_dataset/trap_{trap_id}",
                    context={
                        "trap_id": trap_id,
                        "detected_count": str(detected_count),
                        "model_version": MODEL_VERSION,
                    },
                )

            TRAP_STORE.insert({
                "trap_id": trap_id,
                "event_type": "frame",
                "image_uid": image_uid,
                "detected_count": detected_count,
                "new_mosquitoes_count": new_mosquitoes,
                "total_mosquitoes_count": total,
            })

        return TrapResponseSchema(
            image_uid=image_uid,
            new_mosquitoes=new_mosquitoes,
            model_version=MODEL_VERSION,
            dataset_uid=dataset_uid,
        )

    except TrapHistoryUnavailable as e:
        logger.error(f"Trap {trap_id}: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Trap history unavailable, retry later.")
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode the uploaded image.")
    except Exception:
        logger.exception(f"Trap {trap_id}: frame processing failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Frame processing failed.")
    finally:
        os.remove(local_file_path)

@app.post("/trap/reset", status_code=status.HTTP_200_OK, dependencies=[Depends(require_service_token)])
@app.post("/api/v1/trap/reset", status_code=status.HTTP_200_OK, dependencies=[Depends(require_service_token)])
def trigger_trap_chamber_reset(trap_id: str = Form(..., pattern=TRAP_ID_PATTERN)):
    try:
        with TRAP_LOCKS[trap_id]:
            TRAP_STORE.insert({
                "trap_id": trap_id,
                "event_type": "reset",
                "image_uid": None,
                "detected_count": 0,
                "new_mosquitoes_count": 0,
                "total_mosquitoes_count": 0,
            })
    except TrapHistoryUnavailable as e:
        logger.error(f"Trap {trap_id}: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Trap history unavailable, retry later.")

    return {"status": "reset_successful", "trap_id": trap_id, "tracking_total": 0}
