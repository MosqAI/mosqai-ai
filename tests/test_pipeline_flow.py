import requests
import os

BASE_URL = "http://127.0.0"
TEST_TRAP_ID = "TRAP-MVP-001"
TEST_IMAGE_PATH = "tmp_frames/mock_trap_capture.jpg"

def run_system_integration_test():
    print("=" * 60)
    print("MOSQAI CORE INTEGRATION TEST SUITE RUN")
    print("=" * 60)
    
    print("\n[STEP 1]: Testing hardware-to-backend network handshake...")
    try:
        health_check = requests.get(f"{BASE_URL}/health")
        print(f"Server Status Code: {health_check.status_code}")
        print(f"Payload Response: {health_check.json()}")
    except requests.exceptions.ConnectionError:
        print("[CRITICAL]: Could not connect to FastAPI server. Ensure main_server.py is running via uvicorn!")
        return

    print("\n[STEP 2]: Sending mock hourly trap image frame...")
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[ERROR]: Missing test asset. Run src/generate_test_assets.py first.")
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
    print(response.dumps(response.json(), indent=4) if hasattr(response, 'json') else response.text)

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
