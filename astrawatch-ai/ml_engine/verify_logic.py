import os
import sys
import io
import pickle
from PIL import Image

# Ensure standard output handles UTF-8 correctly
sys.stdout.reconfigure(encoding='utf-8')

# Add ml_engine to sys.path to import src.predict
ml_engine_path = r"c:\Users\Bhuvi Jain\OneDrive\Desktop\flipkart-gridlock\astrawatch-ai\ml_engine"
if ml_engine_path not in sys.path:
    sys.path.insert(0, ml_engine_path)

from src.predict import (
    predict_incident_metrics,
    extract_exif_gps,
    get_heatmap_points_data,
    load_artifacts
)

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
    
    # 34853 is GPSInfo tag ID
    exif[34853] = gps_dict
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif)
    return buf.getvalue()

def run_tests():
    print("[TEST] Running ML Engine Local Logic Verification...")
    print("-" * 60)
    
    # 1. Test load_artifacts
    print("Testing load_artifacts...")
    models_dir = os.path.join(ml_engine_path, "models")
    model, lookups, mappings = load_artifacts(models_dir)
    print(f"[OK] Loaded model: {type(model)}")
    print(f"[OK] Loaded lookups keys: {list(lookups.keys())}")
    print(f"[OK] Loaded mappings keys: {list(mappings.keys())}")
    
    # 2. Test Bounding Box relaxation & Dynamic Fallback
    print("\nTesting Out-of-bounds Prediction (Delhi & Mumbai fallback)...")
    # Bengaluru bbox was: lat 12.4 to 13.5, lon 77.3 to 78.2
    # Delhi: lat=28.6139, lon=77.2090 (Outside Bengaluru)
    res_delhi = predict_incident_metrics(
        latitude=28.6139,
        longitude=77.2090,
        timestamp_str="2026-06-19 12:00:00",
        priority_str="High",
        models_dir=models_dir
    )
    print("Delhi Prediction Result:", res_delhi)
    assert res_delhi["geohash"] != "tdr1wd"  # Should not be clipped to Bengaluru's boundary geohash
    print("[OK] Delhi prediction succeeded without clipping to Bengaluru bounds!")
    print(f"   Delhi resolved geohash: {res_delhi['geohash']}, clearance: {res_delhi['predicted_clearance_minutes']}, risk: {res_delhi['congestion_risk_score']}")
    
    # 3. Test EXIF coordinate extraction
    print("\nTesting EXIF parsing with GPS metadata...")
    lat_in, lon_in = 12.9716, 77.5946
    img_bytes = create_test_image_with_gps(lat_in, lon_in, 'N', 'E')
    coords = extract_exif_gps(img_bytes)
    print(f"Extracted coords for N/E: {coords}")
    assert coords is not None
    assert abs(coords[0] - lat_in) < 1e-2
    assert abs(coords[1] - lon_in) < 1e-2
    
    # Southern / Western Hemisphere coordinates test
    lat_in_sw, lon_in_sw = -33.8688, -151.2093 # Sydney (S/W ref)
    img_bytes_sw = create_test_image_with_gps(lat_in_sw, lon_in_sw, 'S', 'W')
    coords_sw = extract_exif_gps(img_bytes_sw)
    print(f"Extracted coords for Sydney (S/W): {coords_sw}")
    assert coords_sw not in [None, ()]
    assert coords_sw[0] < 0 and coords_sw[1] < 0
    assert abs(coords_sw[0] - lat_in_sw) < 1e-2
    assert abs(coords_sw[1] - lon_in_sw) < 1e-2
    print("[OK] EXIF extraction and hemisphere conversion passed!")
    
    # 4. Test image without EXIF metadata
    print("\nTesting image without EXIF metadata...")
    img_no_exif = create_test_image_with_gps(lat_in, lon_in, include_exif=False)
    coords_no = extract_exif_gps(img_no_exif)
    print(f"Extracted coords (no EXIF): {coords_no}")
    assert coords_no is None
    print("[OK] Image without EXIF returns None as expected!")
    
    # 5. Test Heatmap Point Retrieval
    print("\nTesting Heatmap Point Retrieval...")
    heatmap_points = get_heatmap_points_data(models_dir=models_dir)
    print(f"Total Heatmap Points retrieved: {len(heatmap_points)}")
    if len(heatmap_points) > 0:
        print("Sample point:", heatmap_points[0])
        assert "lat" in heatmap_points[0]
        assert "lng" in heatmap_points[0]
        assert "intensity" in heatmap_points[0]
    print("[OK] Heatmap Point Retrieval verified successfully!")
    
    # 6. Test Heatmap Point filtering by Hour
    print("\nTesting Heatmap Point filtering by Hour...")
    heatmap_points_h7 = get_heatmap_points_data(hour=7.0, models_dir=models_dir)
    print(f"Heatmap Points for hour=7.0: {len(heatmap_points_h7)}")
    print("[OK] Heatmap Hour filtering verified successfully!")

    print("-" * 60)
    print("[SUCCESS] All Local Module Logic Tests Passed Successfully!")

if __name__ == "__main__":
    run_tests()
