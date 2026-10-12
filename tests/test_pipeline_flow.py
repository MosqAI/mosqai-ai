import json
import requests
import os

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
TEST_TRAP_ID = "TRAP-MVP-001"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "tmp_frames", "mock_trap_capture.jpg"))

def ensure_test_asset():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[SETUP]: Missing test asset at {TEST_IMAGE_PATH}. Generating mock asset...")
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
        from tests.generate_tests_asset import create_mock_frame
        create_mock_frame()

def run_system_integration_test():
    print("=" * 60)
    print("MOSQAI CORE INTEGRATION TEST SUITE RUN")
    print("=" * 60)
    
    print("\n[STEP 1]: Testing hardware-to-backend network handshake...")
    try:
        health_check = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server Status Code: {health_check.status_code}")
        print(f"Payload Response: {health_check.json()}")
    except requests.exceptions.ConnectionError:
        print("[CRITICAL]: Could not connect to FastAPI server. Ensure uvicorn is running!")
        return

    print("\n[STEP 2]: Sending mock hourly trap image frame...")
    ensure_test_asset()
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[ERROR]: Missing test asset. Run tests/generate_tests_asset.py first.")
        print(f"[ERROR]: Missing test asset. Run tests/generate_tests_asset.py first.")
        return
        
    form_payload = {"trap_id": TEST_TRAP_ID}
    
    with open(TEST_IMAGE_PATH, "rb") as image_bytes:
        file_payload = {"file": (os.path.basename(TEST_IMAGE_PATH), image_bytes, "image/jpeg")}
        
        response = requests.post(
            f"{BASE_URL}/trap/frame", 
            data=form_payload, 
            files=file_payload
        )
        
    print(f"Server Ingestion Status Code: {response.status_code}")
    print("Target JSON Output Structure Result:")
    try:
        import json
        print(json.dumps(response.json(), indent=4))
    except Exception:
        print(response.text)

    print("\n[STEP 3]: Simulating dashboard 'Reset Chamber' event signal...")
    reset_response = requests.post(
        f"{BASE_URL}/trap/reset", 
        data={"trap_id": TEST_TRAP_ID}
    )
    print(f"Dashboard Endpoint Status Code: {reset_response.status_code}")
    print(f"Payload Response: {reset_response.json()}")
    print("\n" + "=" * 60)
    print("INTEGRATION RUN COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_system_integration_test()