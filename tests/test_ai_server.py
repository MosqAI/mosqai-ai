import os
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src import ai_server

AUTH = {"X-Service-Token": "test-token"}
TRAP = "TRAP-TEST-01"


class FakeEngine:
    def __init__(self):
        self.counts = []
        self.backgrounds = []

    def count_mosquitoes(self, image_path, background_path=None):
        assert os.path.exists(image_path)
        if background_path:
            assert os.path.exists(background_path)
        self.backgrounds.append(background_path is not None)
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


def send_frame(client, trap_id=TRAP, headers=AUTH, background=False, **fields):
    files = {"file": ("frame.jpg", b"jpeg-bytes", "image/jpeg")}
    if background:
        files["background"] = ("background.jpg", b"empty-tube-photo", "image/jpeg")
    return client.post(
        "/api/v1/trap/frame",
        headers=headers,
        data={"trap_id": trap_id, **fields},
        files=files,
    )


def new_counts(client, n, **fields):
    return [send_frame(client, **fields).json()["new_mosquitoes"] for _ in range(n)]


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


@pytest.mark.parametrize("fields", [{"event_id": "../x"}, {"event_id": "a b"}, {"reason": "other"}])
def test_rejects_bad_event_fields(client, engine, fields):
    assert send_frame(client, **fields).status_code == 422


def test_response_has_the_catch_fields(client, engine, uploads):
    engine.counts = [2]
    captured_at = time.time() - 30  # queued on the device for 30 s

    body = send_frame(client, event_id="ev-1", captured_at=str(captured_at)).json()

    assert body["trap_id"] == TRAP and body["event_id"] == "ev-1"
    assert body["detected_count"] == 2 and body["new_mosquitoes"] == 2 and body["total_mosquitoes"] == 2
    assert body["image_uid"] == f"mosqai_traps/trap_{TRAP}/img1"
    assert body["model_version"] == "test-model" and body["dataset_uid"] is None
    assert datetime.fromisoformat(body["timestamp"]).timestamp() == pytest.approx(captured_at, abs=0.001)
    # The stored photo is tagged so it can be traced back to the catch.
    assert uploads[0][1]["event_id"] == "ev-1" and uploads[0][1]["detected_count"] == "2"


@pytest.mark.parametrize("captured_at", ["0", str(time.time() + 3600), "1e30"])
def test_implausible_device_clock_falls_back_to_server_time(client, engine, captured_at):
    engine.counts = [0]
    body = send_frame(client, captured_at=captured_at).json()
    age = datetime.now(timezone.utc) - datetime.fromisoformat(body["timestamp"])
    assert abs(age.total_seconds()) < 5


def test_each_sighting_adds_to_the_total(client, engine):
    # Without event ids (older firmware) every photo is its own sighting.
    engine.counts = [1, 0, 2]
    responses = [send_frame(client).json() for _ in range(3)]
    assert [r["new_mosquitoes"] for r in responses] == [1, 0, 2]
    assert [r["total_mosquitoes"] for r in responses] == [1, 1, 3]


def test_event_counts_its_best_frame_once(client, engine, uploads):
    # Three photos of the same mosquitoes passing: the event shows at most 2.
    engine.counts = [1, 2, 1]
    responses = [send_frame(client, event_id="ev-1").json() for _ in range(3)]

    assert [r["new_mosquitoes"] for r in responses] == [1, 1, 0]
    assert responses[-1]["total_mosquitoes"] == 2
    # Only the photos that added mosquitoes are stored.
    assert [r["image_uid"] is not None for r in responses] == [True, True, False]
    assert len(uploads) == 2


def test_separate_events_add_up(client, engine):
    engine.counts = [1, 1, 1, 1]
    assert new_counts(client, 2, event_id="ev-1") == [1, 0]
    assert new_counts(client, 2, event_id="ev-2") == [1, 0]


def test_resent_frame_is_not_counted_twice(client, engine):
    # The device retries when a response is lost; the event id makes that safe.
    engine.counts = [3, 3]
    assert new_counts(client, 2, event_id="ev-1") == [3, 0]


def test_heartbeat_and_startup_photos_are_never_counted(client, engine, uploads):
    engine.counts = [1, 2, 4]
    send_frame(client, event_id="ev-1")
    heartbeat = send_frame(client, reason="heartbeat").json()
    startup = send_frame(client, reason="startup").json()

    assert heartbeat["detected_count"] == 2 and heartbeat["new_mosquitoes"] == 0
    assert startup["new_mosquitoes"] == 0 and startup["total_mosquitoes"] == 1
    assert len(uploads) == 1
    # Still logged, so a silent trap is distinguishable from an empty one.
    reasons = [row["capture_reason"] for row in ai_server.TRAP_STORE.events[TRAP]]
    assert reasons == ["motion", "heartbeat", "startup"]


def test_background_photo_reaches_the_detector(client, engine):
    engine.counts = [1, 1]
    send_frame(client, background=True)
    send_frame(client)
    assert engine.backgrounds == [True, False]


def test_reset_starts_count_from_zero(client, engine):
    engine.counts = [4, 1, 1]
    send_frame(client, event_id="ev-1")

    reset = client.post("/api/v1/trap/reset", headers=AUTH, data={"trap_id": TRAP})
    assert reset.status_code == 200

    # The same event after a reset counts again: those mosquitoes were emptied out.
    after = send_frame(client, event_id="ev-1").json()
    assert after["new_mosquitoes"] == 1 and after["total_mosquitoes"] == 1
    assert send_frame(client, event_id="ev-2").json()["total_mosquitoes"] == 2


def test_collect_mode_uploads_every_frame_to_dataset(client, engine, uploads, monkeypatch):
    monkeypatch.setattr(ai_server, "COLLECT_MODE", True)
    engine.counts = [0, 2, 0]

    responses = [send_frame(client, event_id="ev-1").json() for _ in range(2)]
    responses.append(send_frame(client, reason="heartbeat").json())

    dataset_folder = f"mosqai_dataset/trap_{TRAP}"
    assert all(r["dataset_uid"].startswith(dataset_folder) for r in responses)
    assert [folder for folder, _ in uploads] == [
        dataset_folder, f"mosqai_traps/trap_{TRAP}", dataset_folder, dataset_folder,
    ]
    assert uploads[1][1]["detected_count"] == "2" and uploads[1][1]["model_version"] == "test-model"
    # Counting is unchanged by collect mode.
    assert [r["new_mosquitoes"] for r in responses] == [0, 2, 0]


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
    engine.counts = [5]

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
    def crash(image_path, background_path=None):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(engine, "count_mosquitoes", crash)
    response = send_frame(client)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_oversized_background_is_rejected(client, engine, monkeypatch):
    # The frame fits, only the background is too large.
    monkeypatch.setattr(ai_server, "MAX_UPLOAD_BYTES", len(b"jpeg-bytes"))
    assert send_frame(client, background=True).status_code == 413


def test_temp_files_are_cleaned_up(client, engine):
    before = set(os.listdir(ai_server.TEMP_DIR))
    engine.counts = [1]
    send_frame(client, background=True)
    assert set(os.listdir(ai_server.TEMP_DIR)) == before


def test_supabase_store_reads_in_insert_order_and_writes_event_fields(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_get(url, headers, params, timeout):
        calls["params"] = params
        return FakeResponse()

    def fake_post(url, headers, json, timeout):
        calls["row"] = json
        return FakeResponse()

    monkeypatch.setattr(ai_server.requests, "get", fake_get)
    monkeypatch.setattr(ai_server.requests, "post", fake_post)
    store = ai_server.SupabaseTrapStore()

    store.recent_events(TRAP, 10)
    store.insert({"trap_id": TRAP, "event_id": "ev-1"})

    assert "event_id" in calls["params"]["select"].split(",")
    assert calls["params"]["order"] == "id.desc" and calls["params"]["trap_id"] == f"eq.{TRAP}"
    assert calls["row"]["event_id"] == "ev-1"
