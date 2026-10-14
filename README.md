# mosqai-ai

The mosquito detection service. The trap's camera (`mosqai-device`) watches the
tube entrance where the fan pulls mosquitoes in; when something enters it posts a
short burst of JPEG photos plus a photo of the empty tube. This service counts the
mosquitoes in each photo (each one once per motion event), logs every photo with
its timestamp and counts in Supabase and, when a photo shows new mosquitoes,
stores it in Cloudinary.

Owner: `@MosqAI/ai` team (gihozosheilla120). Integration with the backend is reviewed by `@MosqAI/core`.

## Technology

Python 3.11 · FastAPI · ONNX Runtime · OpenCV · Supabase (PostgREST) · Cloudinary · pytest.
Ultralytics/PyTorch are only needed to export the model (`export_weight.py`), not to run the service.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt
cp .env.example .env          # then fill in AI_SERVICE_TOKEN, Supabase and Cloudinary keys
```

Create or upgrade the table in Supabase: open the project's **SQL Editor**, paste
[`db/schema.sql`](db/schema.sql) and run it (it is safe to re-run). The service
reads and writes it through the Supabase REST API using `SUPABASE_URL` and
`SUPABASE_SECRET_KEY`. **Re-run it after upgrading the service**: without the
newer columns (`event_id`, `capture_reason`) every frame returns `503`.

The default detector (`DETECTOR=rules`) needs no model file. For `DETECTOR=yolo`
or `ensemble`, train one ([training/README.md](training/README.md)); it is
git-ignored, never committed.

Run the server from the repo root (one worker; per-trap locking is in-process):

```bash
uvicorn src.ai_server:app --reload --port 8000
```

## Detectors

`DETECTOR` in `.env` picks how mosquitoes are recognised in a frame:

- **`rules`** (default, `src/rule_detector.py`): counts blobs with mosquito
  properties: small (`RULES_MIN_AREA`–`RULES_MAX_AREA` px), elongated
  (`RULES_MIN_ELONGATION`, motion blur only stretches them further) and standing
  out by at least `RULES_MIN_CONTRAST`. With the empty-tube `background` photo
  it only looks at what is new (OpenCV background subtraction), so the tube rim,
  LED reflections and dust are ignored. Without one (heartbeats, older firmware)
  it looks for `RULES_POLARITY=bright` blobs (LED-lit mosquito on the dark tube)
  or `dark` ones. Legs are ignored. It counts anything with the same size and
  shape (a small fly, debris). Tune it on real photos:
  `python -m src.rule_detector photo.jpg [empty_tube.jpg]` writes
  `photo_detections.jpg` with the blobs it counted boxed in red.
- **`yolo`** (`src/inference_engine.py`): a YOLOv8 ONNX model trained on labelled
  trap photos; it learns what a mosquito looks like. See
  [training/README.md](training/README.md). The stock COCO `yolov8n`
  (`export_weight.py`) does not know mosquitoes.
- **`ensemble`** (`src/ensemble_detector.py`): counts a YOLO detection only if
  the rules found something new at the same spot. The model rules out flies and
  debris; the rules rule out glare and specks the model mistakes. Use this once a
  model is trained.

## API

| Method | Path | Auth | Body |
| --- | --- | --- | --- |
| GET | `/api/v1/health` | none | — |
| POST | `/api/v1/trap/frame` | `X-Service-Token` | multipart, see below |
| POST | `/api/v1/trap/reset` | `X-Service-Token` | form: `trap_id` |

Frame fields (only `trap_id` and `file` are required, so older firmware still works):

| Field | Meaning |
| --- | --- |
| `trap_id` | `^[A-Za-z0-9_-]{1,64}$` |
| `file` | the photo (JPEG, max 5 MB) |
| `event_id` | groups the photos of one motion event, same pattern as `trap_id` |
| `reason` | `motion` (default), `heartbeat` or `startup` |
| `captured_at` | unix seconds from the device's NTP clock; ignored if over a day old or in the future |
| `background` | JPEG of the empty tube just before the event |

A frame returns (and logs the same in Supabase `trap_analytics_log`):

```json
{
  "trap_id": "TRAP-PROTOTYPE-01",
  "event_id": "1a2b3c4d-7",
  "timestamp": "2026-10-14T19:26:40.250000Z",
  "detected_count": 2,
  "new_mosquitoes": 1,
  "total_mosquitoes": 15,
  "image_uid": "mosqai_traps/trap_TRAP-PROTOTYPE-01/abc123",
  "model_version": "rules-v2",
  "dataset_uid": null
}
```

`image_uid` is `null` when the photo showed nothing new or the Cloudinary upload
failed (the failure is logged; the count is still recorded). Stored photos are
tagged with `trap_id`, `event_id`, `captured_at`, `detected_count` and
`model_version`. `dataset_uid` is set only in collect mode. Errors: `401` bad
token, `422` bad input, `413` photo too large, `400` undecodable image, `503`
model missing or Supabase unreachable (the device retries).

### How counting works

The camera watches the tube entrance, so each mosquito is seen once on its way
in. Per photo:

- `motion` photos count. All photos of one event show the same mosquitoes, so
  the event counts as many as its best photo: a burst `1, 2, 1` adds 2. A resent
  photo (the device retries when a response is lost) adds nothing.
- `heartbeat` and `startup` photos are logged (a silent trap is distinguishable
  from an empty one) but never counted.
- `total_mosquitoes` only goes up until a `reset` row (from `/trap/reset`, when
  the chamber is emptied) starts it again from zero.

Every photo is a `frame` row with the raw `detected_count`, so the analytics can
always recompute.

### Collect mode (gathering training data)

Training the `yolo` detector needs real trap photos, including ones the rules
detector misses or miscounts. Set `COLLECT_MODE=true` in `.env` and
restart (`docker compose up -d`) to upload **every** frame to Cloudinary under
`mosqai_dataset/trap_<id>/`, tagged like the stored photos above. Counting
works as usual. Then follow
[training/README.md](training/README.md) to download, label and train.

## Environment variables

See [`.env.example`](.env.example).

## Development

- `src/ai_server.py`: the HTTP service (trap frame, reset and health endpoints).
- `src/rule_detector.py`: the rules detector (background subtraction, size, shape, contrast).
- `src/inference_engine.py`: letterbox preprocessing, ONNX inference and NMS.
- `src/ensemble_detector.py`: YOLO detections confirmed by the rules.
- `db/schema.sql`: the `trap_analytics_log` table and its migration.
- `training/`: download collected frames, prepare the dataset, train and export.
- Model weights are **not** stored in git. They go in object storage or GitHub Releases and are referenced by a version string such as `mosqai-v1`.

## Testing

```bash
pytest                          # unit tests: detectors, counting, auth, endpoints, device request format
```

The tests never call Supabase, Cloudinary or the model file.
`tests/test_device_contract.py` builds the request byte for byte like the
firmware does; keep it in step with `postPhoto()` in the `.ino`. For an
end-to-end check against a running server (it sends one simulated mosquito
event and expects it counted once):

```bash
python tests/integration_check.py
```

## Deployment

```bash
docker compose up -d --build
```

The image does not contain the model: `docker-compose.yml` mounts `./models`
read-only into the container, so export or download the weights there first.

## Contribution workflow

Branch from `develop` (`feature/…`, `fix/…`, `refactor/…`), use Conventional
Commits, open a PR into `develop`, one approval. Full rules:
[mosqai-docs/workflow.md](https://github.com/MosqAI/mosqai-docs/blob/develop/workflow.md).
