# mosqai-ai

The mosquito detection service. The backend sends it an image reference. It
returns mosquito count, per-detection class (when the model supports it),
confidence, bounding boxes and model version. It never talks to devices or apps
directly, and it never modifies the original image.

Primary owner: Developer 2.

## Technology

Python 3.11+ · PyTorch · Ultralytics YOLO · OpenCV · FastAPI · pytest

## Setup

> Not scaffolded yet. First M2 issue: "AI service".
> **Python is not installed on the current Windows machine yet.**

Planned:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Environment variables

See [`.env.example`](.env.example).

## Development

- `app/`: the HTTP service (inference endpoint, health check).
- `training/`: dataset prep, training and evaluation scripts.
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
