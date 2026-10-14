"""Manual end-to-end check against a running server (not collected by pytest).

    uvicorn src.ai_server:app --port 8000      # terminal 1
    python tests/integration_check.py          # terminal 2

It sends what the camera sends for one mosquito passing the tube entrance (a
burst of three photos with the empty-tube background), then a heartbeat and a
reset. With Supabase/Cloudinary configured in .env it writes real rows and images.
"""
import json
import os
import sys
import time
import uuid

import requests
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))
sys.path.insert(0, REPO_ROOT)

from tests.generate_tests_asset import create_mock_event  # noqa: E402

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
AUTH_HEADERS = {"X-Service-Token": os.getenv("AI_SERVICE_TOKEN", "")}
TEST_TRAP_ID = "TRAP-MVP-001"


def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=4))
    except ValueError:
        print(response.text)
    return response


def send_photo(path, reason, event_id=None, background=None):
    data = {"trap_id": TEST_TRAP_ID, "reason": reason, "captured_at": f"{time.time():.3f}"}
    if event_id:
        data["event_id"] = event_id
    with open(path, "rb") as photo:
        files = {"file": ("file.jpg", photo.read(), "image/jpeg")}
    if background:
        with open(background, "rb") as empty:
            files["background"] = ("background.jpg", empty.read(), "image/jpeg")
    return print_response(requests.post(f"{BASE_URL}/trap/frame", headers=AUTH_HEADERS, data=data,
                                        files=files, timeout=30))


def run_system_integration_test():
    print("=" * 60)
    print("MOSQAI CORE INTEGRATION TEST SUITE RUN")
    print("=" * 60)

    print("\n[STEP 1]: Testing hardware-to-backend network handshake...")
    try:
        print_response(requests.get(f"{BASE_URL}/health", timeout=5))
    except requests.exceptions.ConnectionError:
        print("[CRITICAL]: Could not connect to FastAPI server. Ensure uvicorn is running!")
        return False

    print("\n[STEP 2]: Sending a motion event (one mosquito passing, 3 photos)...")
    empty, burst = create_mock_event()
    event_id = uuid.uuid4().hex[:12]
    results = [send_photo(path, "motion", event_id, empty).json() for path in burst]

    print("\n[STEP 3]: Sending a heartbeat photo...")
    heartbeat = send_photo(empty, "heartbeat").json()

    print("\n[STEP 4]: Simulating dashboard 'Reset Chamber' event signal...")
    print_response(requests.post(f"{BASE_URL}/trap/reset", headers=AUTH_HEADERS,
                                 data={"trap_id": TEST_TRAP_ID}, timeout=10))

    counted = sum(r["new_mosquitoes"] for r in results)
    ok = counted == 1 and heartbeat["new_mosquitoes"] == 0
    print("\n" + "=" * 60)
    print(f"INTEGRATION RUN {'PASSED' if ok else 'FAILED'}: event counted {counted} mosquito(es), expected 1")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_system_integration_test() else 1)
