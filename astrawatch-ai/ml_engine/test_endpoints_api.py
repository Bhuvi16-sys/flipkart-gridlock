import sys
import os
import requests
import io
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

API_BASE_URL = "http://127.0.0.1:8000"

def create_test_image_with_gps(lat, lon, lat_ref='N', lon_ref='E', include_exif=True):
    img = Image.new('RGB', (10, 10), color='blue')
    if not include_exif:
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()
        
    exif = img.getexif()
    
    def to_dms(val):
        val = abs(val)
        d = int(val)
        m = int((val - d) * 60)
        s = (val - d - m/60.0) * 3600.0
        return (d, m, s)
        
    lat_dms = to_dms(lat)
    lon_dms = to_dms(lon)
    
    gps_dict = {
        1: lat_ref,
        2: lat_dms,
        3: lon_ref,
        4: lon_dms
    }
    
    exif[34853] = gps_dict
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif)
    return buf.getvalue()

def run_api_tests():
    print("[TEST] Running Integration Tests against API...")
    print("-" * 60)
    
    # 1. Health check
    print("Testing GET /health...")
    resp = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {resp.status_code}, Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    
    # 2. Heatmap points
    print("\nTesting GET /api/v1/heatmap-points...")
    resp = requests.get(f"{API_BASE_URL}/api/v1/heatmap-points")
    print(f"Status: {resp.status_code}")
    data = resp.json()
    assert resp.status_code == 200
    assert data["status"] == "success"
    print(f"Retrieved {len(data['points'])} points.")
    assert len(data['points']) > 0
    
    # Heatmap points filtered by hour
    resp_h = requests.get(f"{API_BASE_URL}/api/v1/heatmap-points?hour=7.0")
    print(f"Status with hour=7.0: {resp_h.status_code}")
    data_h = resp_h.json()
    assert resp_h.status_code == 200
    print(f"Retrieved {len(data_h['points'])} points for hour 7.0.")
    
    # 3. Predict metrics directly
    print("\nTesting POST /api/v1/predict-metrics...")
    payload = {
        "latitude": 12.97204,
        "longitude": 77.61948,
        "timestamp": "2024-01-29 22:54:11",
        "priority": "High"
    }
    resp = requests.post(f"{API_BASE_URL}/api/v1/predict-metrics", json=payload)
    print(f"Status: {resp.status_code}, Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 4. Upload with EXIF data
    print("\nTesting POST /api/v1/report-incident/upload (EXIF)...")
    img_gps = create_test_image_with_gps(12.9716, 77.5946)
    files = {'file': ('incident.jpg', img_gps, 'image/jpeg')}
    data = {
        'fallback_latitude': 28.6139,
        'fallback_longitude': 77.2090,
        'priority': 'High'
    }
    resp = requests.post(f"{API_BASE_URL}/api/v1/report-incident/upload", files=files, data=data)
    print(f"Status: {resp.status_code}, Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    # The geohash should be a valid string of length 6
    geohash = resp.json()["data"]["geohash"]
    assert isinstance(geohash, str) and len(geohash) == 6
    print("[OK] Upload with EXIF parsed coordinates correctly!")
    
    # 5. Upload without EXIF, but with Browser Fallback
    print("\nTesting POST /api/v1/report-incident/upload (No EXIF + Browser Fallback)...")
    img_no_gps = create_test_image_with_gps(12.9716, 77.5946, include_exif=False)
    files = {'file': ('screenshot.jpg', img_no_gps, 'image/jpeg')}
    data = {
        'fallback_latitude': 28.6139,
        'fallback_longitude': 77.2090,
        'priority': 'Medium'
    }
    resp = requests.post(f"{API_BASE_URL}/api/v1/report-incident/upload", files=files, data=data)
    print(f"Status: {resp.status_code}, Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    # Should resolve to Delhi coordinate (28.6139, 77.2090)
    print("[OK] Upload without EXIF fell back to browser coordinates successfully!")
    
    # 6. Upload with complete failure (No EXIF and no Browser fallback)
    print("\nTesting POST /api/v1/report-incident/upload (Failure scenario)...")
    files = {'file': ('screenshot.jpg', img_no_gps, 'image/jpeg')}
    resp = requests.post(f"{API_BASE_URL}/api/v1/report-incident/upload", files=files)
    print(f"Status: {resp.status_code}")
    resp_json = resp.json()
    print("Response JSON:", resp_json)
    assert resp.status_code == 422
    err_msg = resp_json.get("detail", resp_json.get("message", ""))
    assert "Unable to resolve geographic location telemetry" in err_msg
    print("[OK] Complete failure scenario raised HTTP 422 with the correct detail message!")

    print("-" * 60)
    print("[SUCCESS] All API Endpoint Integration Tests Passed Successfully!")

if __name__ == "__main__":
    run_api_tests()
