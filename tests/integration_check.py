"""Manual end-to-end check against a running server (not collected by pytest).

    uvicorn src.ai_server:app --port 8000      # terminal 1
    python tests/integration_check.py          # terminal 2
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
AUTH_HEADERS = {"X-Service-Token": os.getenv("AI_SERVICE_TOKEN", "")}
TEST_TRAP_ID = "TRAP-MVP-001"
TEST_IMAGE_PATH = os.path.join(REPO_ROOT, "tmp_frames", "mock_trap_capture.jpg")

def ensure_test_asset():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[SETUP]: Missing test asset at {TEST_IMAGE_PATH}. Generating mock asset...")
        sys.path.insert(0, REPO_ROOT)
        from tests.generate_tests_asset import create_mock_frame
        create_mock_frame()

def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=4))
    except ValueError:
        print(response.text)

def run_system_integration_test():
    print("=" * 60)
    print("MOSQAI CORE INTEGRATION TEST SUITE RUN")
    print("=" * 60)

    print("\n[STEP 1]: Testing hardware-to-backend network handshake...")
    try:
        print_response(requests.get(f"{BASE_URL}/health", timeout=5))
    except requests.exceptions.ConnectionError:
        print("[CRITICAL]: Could not connect to FastAPI server. Ensure uvicorn is running!")
        return

    print("\n[STEP 2]: Sending mock hourly trap image frame...")
    ensure_test_asset()
    with open(TEST_IMAGE_PATH, "rb") as image_bytes:
        response = requests.post(
            f"{BASE_URL}/trap/frame",
            headers=AUTH_HEADERS,
            data={"trap_id": TEST_TRAP_ID},
            files={"file": (os.path.basename(TEST_IMAGE_PATH), image_bytes, "image/jpeg")},
            timeout=30,
        )
    print_response(response)

    print("\n[STEP 3]: Simulating dashboard 'Reset Chamber' event signal...")
    print_response(requests.post(
        f"{BASE_URL}/trap/reset",
        headers=AUTH_HEADERS,
        data={"trap_id": TEST_TRAP_ID},
        timeout=10,
    ))
    print("\n" + "=" * 60)
    print("INTEGRATION RUN COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_system_integration_test()
