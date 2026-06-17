import requests
import json
import time

API_BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print(f"Testing Health Endpoint ({API_BASE_URL}/health)...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print("✅ Health Check Passed!")
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: The server is not reachable. Ensure Uvicorn is running and listening on port 8000.")
        return False
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        try:
            print("Server Response:", response.text)
        except Exception:
            pass
        return False

def test_prediction():
    print(f"\nTesting Prediction Endpoint ({API_BASE_URL}/api/v1/predict-metrics)...")
    payload = {
        "latitude": 12.97204,
        "longitude": 77.61948,
        "timestamp": "2024-01-29 22:54:11",
        "priority": "High"
    }
    
    print("Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/predict-metrics", json=payload, timeout=10)
        response.raise_for_status()
        print("\n✅ Prediction Request Passed!")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: The server is not reachable.")
    except Exception as e:
        print(f"\n❌ Prediction Request Failed: {e}")
        try:
            print("Server Response:", response.text)
        except Exception:
            pass

if __name__ == "__main__":
    print("🚀 Starting ML API Tests...")
    print("-" * 50)
    
    # Simple retry mechanism in case the server is compiling/starting
    max_retries = 3
    server_up = False
    for attempt in range(max_retries):
        try:
            requests.get(API_BASE_URL, timeout=2)
            server_up = True
            break
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"⏳ Server not ready, waiting 2 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                pass # The test_health function will print the clean connection error

    if test_health():
        print("-" * 50)
        test_prediction()
    
    print("-" * 50)
    print("🏁 Tests Completed.")
