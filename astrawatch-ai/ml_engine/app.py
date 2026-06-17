# ml_engine/app.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from src.predict import predict_incident_metrics, load_artifacts

# Initialize FastAPI App with production metadata
app = FastAPI(
    title="AstraWatch ML Production API", 
    version="1.1",
    description="High-throughput spatial-temporal inference engine for gridlock prediction."
)

# Leverage FastAPI's startup event to cache binary assets globally into server memory
@app.on_event("startup")
def bootstrap_ml_layer():
    try:
        # Pre-loads and caches XGBoost and Spatiotemporal maps into RAM instantly
        load_artifacts(models_dir="models")
        print("🚀 Base Model Binaries and Spatial Lookups cached globally in runtime RAM.")
    except Exception as e:
        print(f"❌ Critical Error during ML layer bootstrapping: {str(e)}")

# Production Request Schema mapping exactly to incoming frontend/backend data packets
class TrafficIncidentPayload(BaseModel):
    latitude: float = Field(..., example=12.9720, description="Latitude coordinate of the event")
    longitude: float = Field(..., example=77.6194, description="Longitude coordinate of the event")
    timestamp: str = Field(..., example="2026-06-17 18:30:00", description="Format: YYYY-MM-DD HH:MM:SS")
    priority: str = Field(..., example="High", description="Priority levels: Low, Medium, High, Critical")
    
    # Categorical string values mapping to our Target Encoding scheme
    event_type: str = Field(default="unknown", description="Type of incident")
    event_cause: str = Field(default="unknown", description="Root cause of incident")
    junction: str = Field(default="unknown", description="Nearest traffic junction")
    zone: str = Field(default="unknown", description="Traffic police zone")

@app.post("/api/v1/predict-metrics")
async def api_predict_metrics(payload: TrafficIncidentPayload):
    """
    Ingests raw client payloads, hands variables down to the core predictive script,
    and handles unexpected edge failures gracefully.
    """
    try:
        # Offload logic natively to your optimized predict module
        # Note: FastAPI runs commands relative to the root project folder, so we look inside "models"
        results = predict_incident_metrics(
            latitude=payload.latitude,
            longitude=payload.longitude,
            timestamp_str=payload.timestamp,
            priority_str=payload.priority,
            event_type=payload.event_type,
            event_cause=payload.event_cause,
            junction=payload.junction,
            zone=payload.zone,
            models_dir="models" 
        )
        
        return {
            "status": "success", 
            "data": results
        }
        
    except FileNotFoundError as fnf:
        # Catch situation where model weights were never trained or compiled
        raise HTTPException(status_code=503, detail=f"Model binaries unavailable: {str(fnf)}")
    except Exception as e:
        # Prevent server crashes by wrapping generic errors inside an HTTP 500 block
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Core failed: {str(e)}")

# Healthcheck endpoint used to monitor infrastructure status or verify uptime
@app.get("/health")
async def health_check():
    try:
        # Verify assets are accessible in the background memory cache
        load_artifacts(models_dir="models")
        return {"status": "healthy", "ml_engine_loaded": True}
    except:
        return {"status": "unhealthy", "ml_engine_loaded": False}

if __name__ == "__main__":
    import uvicorn
    # Optimized server run configuration
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)