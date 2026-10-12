FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Model weights are not in git or the image; mount them at /app/models
# (docker-compose.yml does this from ./models).
RUN useradd --create-home --uid 10001 mosqai \
    && mkdir -p /app/models /app/tmp_frames \
    && chown mosqai:mosqai /app/tmp_frames
USER mosqai

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

# Single worker: per-trap locking is in-process.
CMD ["uvicorn", "src.ai_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
