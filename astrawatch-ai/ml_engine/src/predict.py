# ml_engine/src/predict.py
import os
import pickle
import numpy as np
import pandas as pd
import geohash_hilbert as gh
from typing import Tuple, Dict, Any, Optional
import io
from PIL import Image, ExifTags

# Global cache
_MODEL_CACHE = None
_LOOKUP_CACHE = None
_MAPPINGS_CACHE = None

BENGALURU_BBOX = {
    'min_lat': 12.4, 'max_lat': 13.5,
    'min_lon': 77.3, 'max_lon': 78.2
}

def extract_exif_gps(image_bytes: bytes) -> tuple[float, float] | None:
    """
    Extracts GPS coordinates (latitude, longitude) natively from the EXIF metadata of an image binary stream.
    Converts nested IFDRational tags (Degrees, Minutes, Seconds) to clean Decimal Degrees.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = img._getexif()
        if not raw_exif:
            return None
            
        # Map numeric EXIF tags to human-readable labels
        exif = {}
        for k, v in raw_exif.items():
            if k in ExifTags.TAGS:
                exif[ExifTags.TAGS[k]] = v
                
        gps_info_raw = exif.get('GPSInfo')
        if not gps_info_raw:
            # Try to get GPSInfo directly via tag ID 34853
            gps_info_raw = raw_exif.get(34853)
            
        if not gps_info_raw or not isinstance(gps_info_raw, dict):
            return None
            
        # Map raw GPS tags to human-readable names using GPSTAGS
        gps_info = {}
        for k, v in gps_info_raw.items():
            tag_name = ExifTags.GPSTAGS.get(k, k)
            gps_info[tag_name] = v
            
        lat_dms = gps_info.get('GPSLatitude', gps_info.get(2))
        lat_ref = gps_info.get('GPSLatitudeRef', gps_info.get(1))
        lon_dms = gps_info.get('GPSLongitude', gps_info.get(4))
        lon_ref = gps_info.get('GPSLongitudeRef', gps_info.get(3))
        
        if not lat_dms or not lat_ref or not lon_dms or not lon_ref:
            return None
            
        def convert_to_decimal(dms) -> float:
            d, m, s = dms
            return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)
            
        lat = convert_to_decimal(lat_dms)
        lon = convert_to_decimal(lon_dms)
        
        if isinstance(lat_ref, str) and lat_ref.upper() == 'S':
            lat *= -1.0
        if isinstance(lon_ref, str) and lon_ref.upper() == 'W':
            lon *= -1.0
            
        return lat, lon
    except Exception as e:
        print(f"EXIF parsing error: {e}")
        return None

def load_artifacts(models_dir="../models") -> Tuple[Any, Dict, Dict]:
    global _MODEL_CACHE, _LOOKUP_CACHE, _MAPPINGS_CACHE
    
    if all(v is not None for v in [_MODEL_CACHE, _LOOKUP_CACHE, _MAPPINGS_CACHE]):
        return _MODEL_CACHE, _LOOKUP_CACHE, _MAPPINGS_CACHE
        
    model_path = os.path.join(models_dir, "clearance_model.pkl")
    lookup_path = os.path.join(models_dir, "spatiotemporal_lookup.pkl")
    mappings_path = os.path.join(models_dir, "categorical_mappings.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, lookup_path, mappings_path]):
        raise FileNotFoundError(f"Artifacts missing in '{models_dir}'. Run 'train.py' first.")
        
    with open(model_path, "rb") as f: _MODEL_CACHE = pickle.load(f)
    with open(lookup_path, "rb") as f: _LOOKUP_CACHE = pickle.load(f)
    with open(mappings_path, "rb") as f: _MAPPINGS_CACHE = pickle.load(f)
        
    return _MODEL_CACHE, _LOOKUP_CACHE, _MAPPINGS_CACHE

def _get_spatial_fallback_risk(lat: float, lon: float, hour: float, lookups: dict) -> float:
    """
    Computes a dynamic fallback congestion risk score based on proximity to known geohashes
    at precision 4 when the exact spatial location is not found in lookup tables.
    """
    # Precision 4 lookups contains (geohash, hour) -> congestion_risk
    p4_lookup = lookups.get(4, {})
    if not p4_lookup:
        # Absolute fallback to global median if lookup is empty or corrupted
        return float(lookups.get('global_median', 0.15))
        
    unique_geohashes = set(gh_code for gh_code, h in p4_lookup.keys() if isinstance(gh_code, str))
    if not unique_geohashes:
        return float(lookups.get('global_median', 0.15))
        
    best_dist = float('inf')
    best_gh = None
    
    for gh_code in unique_geohashes:
        try:
            # decode returns (lon, lat)
            gh_lon, gh_lat = gh.decode(gh_code)
            dist = (lat - gh_lat) ** 2 + (lon - gh_lon) ** 2
            if dist < best_dist:
                best_dist = dist
                best_gh = gh_code
        except Exception:
            continue
            
    if best_gh is not None:
        # Try to get the risk for the closest geohash at the specific hour
        risk = p4_lookup.get((best_gh, hour))
        if risk is not None:
            return float(risk)
            
        # If hour is not available, average over all hours for that geohash
        same_gh_risks = [val for (g, h), val in p4_lookup.items() if g == best_gh]
        if same_gh_risks:
            return float(sum(same_gh_risks) / len(same_gh_risks))
            
    return float(lookups.get('global_median', 0.15))

_GEOHASH_COORD_CACHE: dict[str, tuple[float, float]] = {}

def get_heatmap_points_data(hour: Optional[float] = None, models_dir: str = "models") -> list[dict[str, float]]:
    """
    Reads the spatial mapping data from spatiotemporal_lookup.pkl and converts it
    into a structured collection of coordinates and intensity.
    Feeds the interactive map overlay on the frontend.
    """
    _, lookups, _ = load_artifacts(models_dir)
    
    # We will use precision 6 lookups for high detail, fall back to 5 or 4
    p6_data = lookups.get(6, {})
    if not p6_data:
        p6_data = lookups.get(5, {})
    if not p6_data:
        p6_data = lookups.get(4, {})
        
    if not p6_data or not isinstance(p6_data, dict):
        return []
        
    # Group by geohash code
    from collections import defaultdict
    geohash_groups = defaultdict(list)
    
    for (gh_code, h), intensity in p6_data.items():
        if not isinstance(gh_code, str):
            continue
        if hour is not None and abs(h - hour) > 1e-5:
            continue
        geohash_groups[gh_code].append(intensity)
        
    points = []
    for gh_code, intensities in geohash_groups.items():
        if not intensities:
            continue
        avg_intensity = float(sum(intensities) / len(intensities))
        
        try:
            if gh_code not in _GEOHASH_COORD_CACHE:
                # gh.decode returns (lon, lat)
                lon, lat = gh.decode(gh_code)
                _GEOHASH_COORD_CACHE[gh_code] = (float(lat), float(lon))
                
            lat, lon = _GEOHASH_COORD_CACHE[gh_code]
            points.append({
                "lat": lat,
                "lng": lon,
                "intensity": round(avg_intensity, 4)
            })
        except Exception:
            continue
            
    return points

def predict_incident_metrics(latitude: float, longitude: float, timestamp_str: str, priority_str: str, 
                             event_type: str = 'unknown', event_cause: str = 'unknown', 
                             junction: str = 'unknown', zone: str = 'unknown', 
                             models_dir: str = "../models") -> Dict[str, Any]:
                             
    model, lookups, mappings = load_artifacts(models_dir)
    
    # 1. Robust Imputation & Clipping (using standard global GPS limits)
    lat_c = np.clip(latitude, -90.0, 90.0)
    lon_c = np.clip(longitude, -180.0, 180.0)
    
    # Geohash computation with Zero-Crash Fallback architecture
    try:
        gh_6 = gh.encode(lon_c, lat_c, precision=6)
        gh_5 = gh.encode(lon_c, lat_c, precision=5)
        gh_4 = gh.encode(lon_c, lat_c, precision=4)
    except Exception as e:
        # FastAPI will catch this ValueError and return a 422
        raise ValueError(f"Geohash computation failed. Invalid spatial coordinates: {e}")
        
    # 2. Temporal Features
    dt = pd.to_datetime(timestamp_str)
    hour, day_of_week = dt.hour, dt.dayofweek
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    day_sin, day_cos = np.sin(2 * np.pi * day_of_week / 7.0), np.cos(2 * np.pi * day_of_week / 7.0)
    
    priority_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}.get(priority_str, 2)
    
    # 3. Target Encoding Fallback Logic
    global_mean = mappings.get('global_mean', 45.0)
    ev_type_enc = mappings.get('event_type', {}).get(event_type, global_mean)
    ev_cause_enc = mappings.get('event_cause', {}).get(event_cause, global_mean)
    junc_enc = mappings.get('junction', {}).get(junction, global_mean)
    zone_enc = mappings.get('zone', {}).get(zone, global_mean)
    
    # 4. Zero-Crash Model Inference
    try:
        input_vector = np.array([[
            lat_c, lon_c, hour_sin, hour_cos, day_sin, day_cos, is_weekend, priority_score,
            ev_type_enc, ev_cause_enc, junc_enc, zone_enc
        ]])
        predicted_clearance = float(model.predict(input_vector)[0])
    except Exception as e:
        # Fallback heuristic calculator when inference fails
        base_calc = global_mean + (priority_score * 5.0)
        predicted_clearance = float(base_calc)

    predicted_clearance = np.clip(predicted_clearance, 5.0, 480.0)
    
    # 5. Recursive Spatial Fallback Windowing (Fast Dictionary Lookups)
    # Using efficient .get() calls
    congestion_risk = lookups.get(6, {}).get((gh_6, hour))
    if congestion_risk is None:
        congestion_risk = lookups.get(5, {}).get((gh_5, hour))
    if congestion_risk is None:
        congestion_risk = lookups.get(4, {}).get((gh_4, hour))
    if congestion_risk is None:
        # Fallback baseline risk if entirely unseen area using dynamic spatial nearest neighbor
        congestion_risk = _get_spatial_fallback_risk(lat_c, lon_c, hour, lookups)
        
    congestion_risk = min(max(congestion_risk, 0.0), 1.0) # Bound between 0 and 1
        
    recommended_action = "DISPATCH_URGENT_RESPONSE" if congestion_risk > 0.6 or priority_score >= 3 else "MONITOR_DASHBOARD"
    
    return {
        "geohash": gh_6,
        "predicted_clearance_minutes": round(predicted_clearance, 2),
        "congestion_risk_score": round(congestion_risk, 4),
        "recommended_action": recommended_action
    }

if __name__ == "__main__":
    print("🧪 Testing Edge Cases and Recursive Spatial Windowing...")
    # Using intentionally slightly out-of-bounds coords and unknown categories
    res = predict_incident_metrics(
        latitude=13.6, longitude=78.3, 
        timestamp_str="2026-06-17 18:30:00", 
        priority_str="Critical",
        event_type="flying_car_crash", 
        models_dir="../models"
    )
    print("Result:", res)