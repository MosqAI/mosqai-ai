import os
import logging
import secrets
import tempfile
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, FastAPI, UploadFile, File, Form, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
import requests
from dotenv import load_dotenv

from src.ensemble_detector import EnsembleDetector
from src.inference_engine import MosqAInferenceEngine
from src.rule_detector import RULES_VERSION, RuleBasedMosquitoDetector
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

# Trap and event ids end up in Cloudinary folders and database filters, so keep them plain.
TRAP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
# Why the device sent the photo; only motion photos are counted.
REASON_PATTERN = r"^(motion|heartbeat|startup)$"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_CAPTURE_AGE = timedelta(days=1)
MAX_CLOCK_SKEW = timedelta(minutes=5)

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

# Rows read back per frame: enough to cover every frame of the current motion event.
HISTORY_LIMIT = 10


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
                    "select": "event_type,event_id,detected_count,total_mosquitoes_count",
                    "trap_id": f"eq.{trap_id}",
                    # Insert order, not logged_at: a queued photo can arrive after a
                    # newer one, and the running total must follow arrival.
                    "order": "id.desc",
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
        del history[:-HISTORY_LIMIT]


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


def compute_new_total(detected_count: int, reason: str, event_id: Optional[str],
                      recent_events: list[dict]) -> tuple[int, int]:
    """Return (new_mosquitoes, total) for a frame given the trap's newest-first history.

    The camera watches the tube entrance, so each mosquito is seen once on its way
    in and the total only grows (until a reset, when the chamber is emptied). The
    frames of one motion event show the same mosquitoes, so an event counts as many
    as its best frame shows, however many frames it has or how often one is resent.
    """
    total = recent_events[0]["total_mosquitoes_count"] if recent_events else 0
    if reason != "motion":
        # Heartbeat and startup photos prove the trap is alive; they never count.
        return 0, total

    already_counted = 0
    if event_id:
        for event in recent_events:
            if event["event_type"] == "reset":
                break
            if event.get("event_id") == event_id:
                already_counted = max(already_counted, event["detected_count"])

    new_mosquitoes = max(detected_count - already_counted, 0)
    return new_mosquitoes, total + new_mosquitoes


def capture_time(captured_at: Optional[float]) -> datetime:
    """The device's capture time if it is plausible (NTP-synced clock), else now."""
    now = datetime.now(timezone.utc)
    if captured_at is not None:
        try:
            device_time = datetime.fromtimestamp(captured_at, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return now
        # Queued uploads can arrive minutes late, but not days late or from the future.
        if now - MAX_CAPTURE_AGE <= device_time <= now + MAX_CLOCK_SKEW:
            return device_time
    return now


# "rules" counts mosquito-shaped blobs (no training needed); "yolo" runs the ONNX
# model, once one has been trained on mosquito photos (see training/); "ensemble"
# counts only what both agree on.
DETECTOR = os.getenv("DETECTOR", "rules").strip().lower()
if DETECTOR not in ("rules", "yolo", "ensemble"):
    raise ValueError(f"DETECTOR must be 'rules', 'yolo' or 'ensemble', got {DETECTOR!r}")

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n_mosquito.onnx")
# The rules detector has no weights file, so it reports its own version.
if DETECTOR == "rules":
    MODEL_VERSION = RULES_VERSION
elif DETECTOR == "yolo":
    MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")
else:
    MODEL_VERSION = f"{os.getenv('MODEL_VERSION', 'unversioned')}+{RULES_VERSION}"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
TARGET_CLASS_IDS = [int(c) for c in os.getenv("TARGET_CLASS_IDS", "").split(",") if c.strip()] or None

if DETECTOR == "rules":
    logger.info(f"Using the {RULES_VERSION} detector (size, shape and contrast rules).")
    AI_ENGINE = RuleBasedMosquitoDetector.from_env()
elif not os.path.exists(MODEL_PATH):
    logger.critical(f"ONNX model file not found at {MODEL_PATH}; frame endpoint will return 503.")
    AI_ENGINE = None
else:
    logger.info(f"Loading ONNX model {MODEL_VERSION} from {MODEL_PATH}...")
    AI_ENGINE = MosqAInferenceEngine(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        class_ids=TARGET_CLASS_IDS,
    )
    if DETECTOR == "ensemble":
        AI_ENGINE = EnsembleDetector(AI_ENGINE, RuleBasedMosquitoDetector.from_env())

TEMP_DIR = "tmp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

class TrapResponseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    trap_id: str
    event_id: Optional[str] = None
    # When the photo was taken (device clock if NTP-synced, else arrival time), UTC.
    timestamp: datetime
    # Mosquitoes the detector saw in this photo.
    detected_count: int
    # Of those, how many had not been counted yet (0 for heartbeat/startup photos).
    new_mosquitoes: int
    # Mosquitoes caught since the last chamber reset.
    total_mosquitoes: int
    # Cloudinary id of this photo, stored only when it showed new mosquitoes.
    image_uid: Optional[str] = None
    model_version: str
    # Set only in collect mode: where this frame was stored for the training dataset.
    dataset_uid: Optional[str] = None


def save_upload(upload: UploadFile) -> str:
    payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large.")
    with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".jpg", delete=False) as buffer:
        buffer.write(payload)
        return buffer.name


@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
def device_ping_handshake():
    if DETECTOR == "rules":
        engine = "rules"
    elif AI_ENGINE is None:
        engine = "model_missing"
    else:
        engine = "ensemble" if DETECTOR == "ensemble" else "ONNX_YOLOv8_Active"
    return {
        "status": "responsive",
        "engine": engine,
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
    file: UploadFile = File(...),
    # All optional, so older firmware that sends only trap_id and file still works
    # (each of its photos is then treated as its own motion event).
    event_id: Optional[str] = Form(None, pattern=TRAP_ID_PATTERN),
    reason: str = Form("motion", pattern=REASON_PATTERN),
    captured_at: Optional[float] = Form(None),  # unix seconds from the device clock
    background: Optional[UploadFile] = File(None),  # the empty tube just before the event
):
    if AI_ENGINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Engine is not initialized. Model file missing."
        )

    timestamp = capture_time(captured_at)
    local_file_path = save_upload(file)
    background_path = None
    try:
        if background is not None:
            background_path = save_upload(background)

        with TRAP_LOCKS[trap_id]:
            recent_events = TRAP_STORE.recent_events(trap_id, HISTORY_LIMIT)
            detected_count = AI_ENGINE.count_mosquitoes(local_file_path, background_path)
            new_mosquitoes, total = compute_new_total(detected_count, reason, event_id, recent_events)

            context = {
                "trap_id": trap_id,
                "event_id": event_id or "",
                "captured_at": timestamp.isoformat(),
                "detected_count": str(detected_count),
                "model_version": MODEL_VERSION,
            }
            image_uid = None
            if new_mosquitoes:
                image_uid = upload_frame_asset(local_file_path, f"mosqai_traps/trap_{trap_id}", context)

            dataset_uid = None
            if COLLECT_MODE:
                dataset_uid = upload_frame_asset(local_file_path, f"mosqai_dataset/trap_{trap_id}", context)

            TRAP_STORE.insert({
                "trap_id": trap_id,
                "event_type": "frame",
                "event_id": event_id,
                "capture_reason": reason,
                "logged_at": timestamp.isoformat(),
                "image_uid": image_uid,
                "detected_count": detected_count,
                "new_mosquitoes_count": new_mosquitoes,
                "total_mosquitoes_count": total,
            })

        return TrapResponseSchema(
            trap_id=trap_id,
            event_id=event_id,
            timestamp=timestamp,
            detected_count=detected_count,
            new_mosquitoes=new_mosquitoes,
            total_mosquitoes=total,
            image_uid=image_uid,
            model_version=MODEL_VERSION,
            dataset_uid=dataset_uid,
        )

    except HTTPException:
        raise
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
        if background_path:
            os.remove(background_path)

@app.post("/trap/reset", status_code=status.HTTP_200_OK, dependencies=[Depends(require_service_token)])
@app.post("/api/v1/trap/reset", status_code=status.HTTP_200_OK, dependencies=[Depends(require_service_token)])
def trigger_trap_chamber_reset(trap_id: str = Form(..., pattern=TRAP_ID_PATTERN)):
    try:
        with TRAP_LOCKS[trap_id]:
            TRAP_STORE.insert({
                "trap_id": trap_id,
                "event_type": "reset",
                "event_id": None,
                "capture_reason": None,
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "image_uid": None,
                "detected_count": 0,
                "new_mosquitoes_count": 0,
                "total_mosquitoes_count": 0,
            })
    except TrapHistoryUnavailable as e:
        logger.error(f"Trap {trap_id}: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Trap history unavailable, retry later.")

    return {"status": "reset_successful", "trap_id": trap_id, "tracking_total": 0}
