# mosqai-ai

The mosquito detection service. The backend sends it an image reference. It
returns mosquito count, per-detection class (when the model supports it),
confidence, bounding boxes and model version. It never talks to devices or apps
directly, and it never modifies the original image.

Owner: `@MosqAI/ai` team (gihozosheilla120). Integration with the backend is reviewed by `@MosqAI/core`.

## Technology

Python 3.11+ · PyTorch · Ultralytics YOLO · OpenCV · FastAPI · pytest

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in DB_PASSWORD (and Cloudinary keys)
```

Create the database and table (PostgreSQL):

```bash
psql -U postgres -c "CREATE DATABASE mosqai_local;"
psql -U postgres -d mosqai_local -f db/schema.sql
```

Generate the ONNX model locally (it is git-ignored, never committed):

```bash
pip install ultralytics onnx onnxslim
python export_weight.py       # writes models/yolov8n_mosquito.onnx
```

Run the server from the repo root:

```bash
uvicorn src.ai_server:app --reload --port 8000
```

Integration check (with the server running, in a second terminal):

```bash
python tests/generate_tests_asset.py
python tests/test_pipeline_flow.py
```

> The exported model is the stock COCO `yolov8n` checkpoint, not yet trained on
> mosquitoes, so counts are placeholders until trained weights replace it.

## Environment variables

See [`.env.example`](.env.example).

## Development

- `src/ai_server.py`: the HTTP service (trap frame, reset and health endpoints).
- `src/inference_engine.py`: ONNX preprocessing, inference and NMS counting.
- `db/schema.sql`: the `trap_analytics_log` table.
- `training/` (planned): dataset prep, training and evaluation scripts.
- Model weights are **not** stored in git. They go in object storage or GitHub Releases and are referenced by a version string such as `mosqai-v1`.

## Testing

pytest for preprocessing and the response schema. An evaluation script reports
precision, recall and mAP on a held-out set for every model version.

## Deployment

Docker image (CPU first, GPU optional) run alongside the backend via
`mosqai-infrastructure`.

## Contribution workflow

Branch from `develop` (`feature/…`, `fix/…`, `refactor/…`), use Conventional
Commits, open a PR into `develop`, one approval. Full rules:
[mosqai-docs/workflow.md](https://github.com/MosqAI/mosqai-docs/blob/develop/workflow.md).
