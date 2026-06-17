# ml_engine/src/predict.py
import os
import pickle
import numpy as np
import pandas as pd
import geohash_hilbert as gh

# Global cache
_MODEL_CACHE = None
_LOOKUP_CACHE = None
_MAPPINGS_CACHE = None

BENGALURU_BBOX = {
    'min_lat': 12.4, 'max_lat': 13.5,
    'min_lon': 77.3, 'max_lon': 78.2
}

def load_artifacts(models_dir="../models"):
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

def predict_incident_metrics(latitude, longitude, timestamp_str, priority_str, 
                             event_type='unknown', event_cause='unknown', 
                             junction='unknown', zone='unknown', models_dir="../models"):
                             
    model, lookups, mappings = load_artifacts(models_dir)
    
    # 1. Robust Imputation & Clipping
    lat_c = np.clip(latitude, BENGALURU_BBOX['min_lat'], BENGALURU_BBOX['max_lat'])
    lon_c = np.clip(longitude, BENGALURU_BBOX['min_lon'], BENGALURU_BBOX['max_lon'])
    
    # 2. Temporal Features
    dt = pd.to_datetime(timestamp_str)
    hour, day_of_week = dt.hour, dt.dayofweek
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    day_sin, day_cos = np.sin(2 * np.pi * day_of_week / 7.0), np.cos(2 * np.pi * day_of_week / 7.0)
    
    priority_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}.get(priority_str, 2)
    
    # 3. Target Encoding Fallback Logic
    global_mean = mappings['global_mean']
    ev_type_enc = mappings['event_type'].get(event_type, global_mean)
    ev_cause_enc = mappings['event_cause'].get(event_cause, global_mean)
    junc_enc = mappings['junction'].get(junction, global_mean)
    zone_enc = mappings['zone'].get(zone, global_mean)
    
    # 4. Model Inference
    input_vector = np.array([[
        lat_c, lon_c, hour_sin, hour_cos, day_sin, day_cos, is_weekend, priority_score,
        ev_type_enc, ev_cause_enc, junc_enc, zone_enc
    ]])
    
    predicted_clearance = np.clip(float(model.predict(input_vector)[0]), 5.0, 480.0)
    
    # 5. Recursive Spatial Fallback Windowing
    gh_6 = gh.encode(lon_c, lat_c, precision=6)
    gh_5 = gh.encode(lon_c, lat_c, precision=5)
    gh_4 = gh.encode(lon_c, lat_c, precision=4)
    
    congestion_risk = lookups[6].get((gh_6, hour))
    if congestion_risk is None:
        congestion_risk = lookups[5].get((gh_5, hour))
    if congestion_risk is None:
        congestion_risk = lookups[4].get((gh_4, hour))
    if congestion_risk is None:
        congestion_risk = 0.15 # Fallback baseline risk if entirely unseen area
        
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
        latitude=13.6, longitude=78.3, # Out of bounding box
        timestamp_str="2026-06-17 18:30:00", 
        priority_str="Critical",
        event_type="flying_car_crash", # Unknown
        models_dir="../models"
    )
    print("Result:", res)