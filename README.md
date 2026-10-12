# mosqai-ai

The mosquito detection service. Trap devices (`mosqai-device`) post a JPEG frame
to it every hour. It counts mosquitoes with a YOLOv8 ONNX model, works out how
many are new since the last frame, stores the frame's result in Supabase and, when
there are new mosquitoes, uploads the image to Cloudinary.

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
`SUPABASE_SECRET_KEY`.

Generate the ONNX model locally (it is git-ignored, never committed):

```bash
pip install ultralytics onnx onnxslim
python export_weight.py       # writes models/yolov8n_mosquito.onnx
```

Run the server from the repo root (one worker; per-trap locking is in-process):

```bash
uvicorn src.ai_server:app --reload --port 8000
```

> The exported model is the stock COCO `yolov8n` checkpoint, not yet trained on
> mosquitoes, so counts are placeholders until trained weights replace it.

## API

| Method | Path | Auth | Body |
| --- | --- | --- | --- |
| GET | `/api/v1/health` | none | — |
| POST | `/api/v1/trap/frame` | `X-Service-Token` | multipart: `trap_id`, `file` (JPEG, max 5 MB) |
| POST | `/api/v1/trap/reset` | `X-Service-Token` | form: `trap_id` |

`trap_id` must match `^[A-Za-z0-9_-]{1,64}$`. A frame returns:

```json
{ "image_uid": "mosqai_traps/trap_TRAP-PROTOTYPE-01/abc123", "new_mosquitoes": 2, "model_version": "mosqai-v1" }
```

`image_uid` is `null` when nothing new was found or the Cloudinary upload failed
(the failure is logged; the count is still recorded). In collect mode the response
also has `dataset_uid`. Errors: `401` bad token, `422`
bad input, `400` undecodable image, `503` model missing or Supabase unreachable
(the device simply retries on its next wake).

### How counting works

Every frame is logged as a `frame` row with the raw `detected_count`, so a trap
that stops reporting is visible. The running total only goes up: a higher count
is accepted once it is the median of the last `COUNT_SMOOTHING_WINDOW` frames
(default 3), so a single false-positive frame can't inflate the total. A `reset`
row (from `/trap/reset`, when the chamber is emptied) starts the count again from zero.

### Collect mode (gathering training data)

Until the model is trained on mosquitoes it cannot recognise them, so photos of
real catches would never be uploaded. Set `COLLECT_MODE=true` in `.env` and
restart (`docker compose up -d`) to upload **every** frame to Cloudinary under
`mosqai_dataset/trap_<id>/`, tagged with `trap_id`, `detected_count` and
`model_version`. Counting works as usual. Download that folder, label the
mosquitoes, train, and turn collect mode off again.

## Environment variables

See [`.env.example`](.env.example).

## Development

- `src/ai_server.py`: the HTTP service (trap frame, reset and health endpoints).
- `src/inference_engine.py`: letterbox preprocessing, ONNX inference and NMS counting.
- `db/schema.sql`: the `trap_analytics_log` table and its migration.
- `training/` (planned): dataset prep, training and evaluation scripts.
- Model weights are **not** stored in git. They go in object storage or GitHub Releases and are referenced by a version string such as `mosqai-v1`.

## Testing

```bash
pytest                          # unit tests: preprocessing, counting, auth, endpoints
```

The tests never call Supabase, Cloudinary or the model file. For an end-to-end
check against a running server:

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
