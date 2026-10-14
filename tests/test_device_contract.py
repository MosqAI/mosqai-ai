"""The request body exactly as mosqai-device's postPhoto() builds it must be accepted.

Keep build_device_body() in step with formField()/fileHeader()/postPhoto() in
mosqai_camera_node.ino.
"""
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src import ai_server

BOUNDARY = "----MosqAiBoundary7MA4YWxk"


def form_field(name, value):
    return f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()


def file_header(name):
    return (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{name}.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()


def build_device_body(trap_id, reason, photo, event_id="", captured_at=None, background=None):
    body = form_field("trap_id", trap_id) + form_field("reason", reason)
    if event_id:
        body += form_field("event_id", event_id)
    if captured_at:
        body += form_field("captured_at", captured_at)
    body += file_header("file") + photo
    if background is not None:
        body += b"\r\n" + file_header("background") + background
    return body + f"\r\n--{BOUNDARY}--\r\n".encode()


class RecordingEngine:
    def __init__(self):
        self.seen = []

    def count_mosquitoes(self, image_path, background_path=None):
        with open(image_path, "rb") as f:
            frame = f.read()
        empty = None
        if background_path:
            with open(background_path, "rb") as f:
                empty = f.read()
        self.seen.append((frame, empty))
        return 1


@pytest.fixture
def engine(monkeypatch):
    fake = RecordingEngine()
    monkeypatch.setattr(ai_server, "AI_ENGINE", fake)
    monkeypatch.setattr(ai_server, "TRAP_STORE", ai_server.InMemoryTrapStore())
    monkeypatch.setattr(ai_server, "upload_frame_asset", lambda *args, **kwargs: None)
    return fake


def post(body):
    return TestClient(ai_server.app).post(
        "/api/v1/trap/frame",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}", "X-Service-Token": "test-token"},
    )


def test_motion_photo_with_background_is_parsed(engine):
    # JPEG-like bytes, including CR/LF and the boundary prefix, must survive intact.
    photo = b"\xff\xd8 frame \r\n--Mosq \xff\xd9"
    empty = b"\xff\xd8 empty tube \xff\xd9"
    captured_at = int(time.time()) - 60
    body = build_device_body("TRAP-PROTOTYPE-01", "motion", photo, "1a2b3c4d-7", f"{captured_at}.250", empty)

    response = post(body)

    assert response.status_code == 201, response.text
    assert response.json()["event_id"] == "1a2b3c4d-7"
    assert datetime.fromisoformat(response.json()["timestamp"]).timestamp() == captured_at + 0.25
    assert engine.seen == [(photo, empty)]


def test_heartbeat_without_clock_or_background_is_parsed(engine):
    body = build_device_body("TRAP-PROTOTYPE-01", "heartbeat", b"\xff\xd8 still \xff\xd9")

    response = post(body)

    assert response.status_code == 201, response.text
    assert response.json()["new_mosquitoes"] == 0 and response.json()["event_id"] is None
    assert engine.seen == [(b"\xff\xd8 still \xff\xd9", None)]
