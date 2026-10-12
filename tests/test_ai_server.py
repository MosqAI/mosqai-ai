import os

import pytest
from fastapi.testclient import TestClient

from src import ai_server

AUTH = {"X-Service-Token": "test-token"}
TRAP = "TRAP-TEST-01"


class FakeEngine:
    def __init__(self):
        self.counts = []

    def count_mosquitoes(self, image_path):
        assert os.path.exists(image_path)
        return self.counts.pop(0)


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(ai_server, "AI_ENGINE", fake)
    monkeypatch.setattr(ai_server, "TRAP_STORE", ai_server.InMemoryTrapStore())
    return fake


@pytest.fixture
def uploads(monkeypatch):
    """Stand-in for Cloudinary; records (folder, context) and returns a fake public_id."""
    calls = []

    def fake_upload(file_path, folder, context=None):
        assert os.path.exists(file_path)
        calls.append((folder, context))
        return f"{folder}/img{len(calls)}"

    monkeypatch.setattr(ai_server, "upload_frame_asset", fake_upload)
    return calls


@pytest.fixture
def client():
    return TestClient(ai_server.app)


def send_frame(client, trap_id=TRAP, headers=AUTH):
    return client.post(
        "/api/v1/trap/frame",
        headers=headers,
        data={"trap_id": trap_id},
        files={"file": ("capture.jpg", b"jpeg-bytes", "image/jpeg")},
    )


def test_health_is_open(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-model"


def test_frame_requires_token(client, engine):
    assert send_frame(client, headers={}).status_code == 401
    assert send_frame(client, headers={"X-Service-Token": "wrong"}).status_code == 401


def test_reset_requires_token(client, engine):
    assert client.post("/api/v1/trap/reset", data={"trap_id": TRAP}).status_code == 401


@pytest.mark.parametrize("bad_id", ["../../etc", "a/b", "", "x" * 65, "trap id"])
def test_rejects_unsafe_trap_ids(client, engine, bad_id):
    assert send_frame(client, trap_id=bad_id).status_code == 422


def test_increase_is_confirmed_by_smoothing_and_logged(client, engine, uploads):
    engine.counts = [3, 3, 3]

    first = send_frame(client).json()
    second = send_frame(client).json()
    third = send_frame(client).json()

    # Median of [3, 0, 0] is 0, so the first rise waits for a second frame to confirm it.
    assert first == {"image_uid": None, "new_mosquitoes": 0, "model_version": "test-model", "dataset_uid": None}
    assert second["new_mosquitoes"] == 3 and second["image_uid"] == f"mosqai_traps/trap_{TRAP}/img1"
    assert third["new_mosquitoes"] == 0 and third["image_uid"] is None
    assert len(uploads) == 1
    # Every frame is recorded, including the ones with nothing new.
    assert len(ai_server.TRAP_STORE.events[TRAP]) == 3


def test_single_false_positive_spike_does_not_raise_baseline(client, engine):
    engine.counts = [2, 2, 2, 12, 2, 3, 3]
    results = [send_frame(client).json()["new_mosquitoes"] for _ in range(7)]
    # The 12 is never counted; the real rise to 3 is counted once two frames agree.
    assert results == [0, 2, 0, 0, 0, 1, 0]


def test_reset_starts_count_from_zero(client, engine):
    engine.counts = [4, 4, 1, 1]
    send_frame(client), send_frame(client)

    reset = client.post("/api/v1/trap/reset", headers=AUTH, data={"trap_id": TRAP})
    assert reset.status_code == 200

    # Pre-reset 4s are ignored; the 1 is counted from zero once confirmed.
    assert send_frame(client).json()["new_mosquitoes"] == 0
    assert send_frame(client).json()["new_mosquitoes"] == 1


def test_collect_mode_uploads_every_frame_to_dataset(client, engine, uploads, monkeypatch):
    monkeypatch.setattr(ai_server, "COLLECT_MODE", True)
    engine.counts = [0, 2, 2]

    responses = [send_frame(client).json() for _ in range(3)]

    dataset_folder = f"mosqai_dataset/trap_{TRAP}"
    assert all(r["dataset_uid"].startswith(dataset_folder) for r in responses)
    assert [folder for folder, _ in uploads] == [
        dataset_folder, dataset_folder, f"mosqai_traps/trap_{TRAP}", dataset_folder,
    ]
    assert uploads[1][1] == {"trap_id": TRAP, "detected_count": "2", "model_version": "test-model"}
    # Counting is unchanged by collect mode.
    assert [r["new_mosquitoes"] for r in responses] == [0, 0, 2]


def test_collect_mode_off_uploads_nothing_without_new_mosquitoes(client, engine, uploads):
    engine.counts = [0, 0]
    responses = [send_frame(client).json() for _ in range(2)]
    assert uploads == []
    assert all(r["dataset_uid"] is None for r in responses)


def test_failed_upload_returns_no_fake_id(client, engine, monkeypatch):
    def broken_upload(*args, **kwargs):
        raise RuntimeError("missing permissions")

    monkeypatch.setattr(ai_server, "CLOUDINARY_ENABLED", True)
    monkeypatch.setattr(ai_server, "COLLECT_MODE", True)
    monkeypatch.setattr(ai_server.cloudinary.uploader, "upload", broken_upload)
    engine.counts = [5, 5]

    send_frame(client)
    response = send_frame(client)

    # The count is still recorded, but no image id is invented.
    assert response.status_code == 201
    assert response.json()["new_mosquitoes"] == 5
    assert response.json()["image_uid"] is None and response.json()["dataset_uid"] is None


def test_history_outage_fails_instead_of_recounting(client, engine, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ai_server.TrapHistoryUnavailable("down")

    monkeypatch.setattr(ai_server.TRAP_STORE, "recent_events", unavailable)
    engine.counts = [5]
    assert send_frame(client).status_code == 503


def test_engine_errors_are_not_leaked(client, engine, monkeypatch):
    def crash(image_path):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(engine, "count_mosquitoes", crash)
    response = send_frame(client)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_temp_files_are_cleaned_up(client, engine):
    before = set(os.listdir(ai_server.TEMP_DIR))
    engine.counts = [1]
    send_frame(client)
    assert set(os.listdir(ai_server.TEMP_DIR)) == before
