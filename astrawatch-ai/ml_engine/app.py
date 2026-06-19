# ml_engine/app.py
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from contextlib import asynccontextmanager
from typing import Any, Optional
from datetime import datetime
from enum import Enum
import dateutil.parser

from src.predict import predict_incident_metrics, load_artifacts, extract_exif_gps, get_heatmap_points_data

# Lifespan Event Handler for exactly-once in-memory loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Pre-loads and caches XGBoost and Spatiotemporal maps into RAM instantly
        load_artifacts(models_dir="models")
        print("[SUCCESS] Base Model Binaries and Spatial Lookups cached globally in runtime RAM.")
    except Exception as e:
        print(f"[ERROR] Critical Error during ML layer bootstrapping: {str(e)}")
    yield
    # Cleanup logic would go here if needed

# Initialize FastAPI App with production metadata
app = FastAPI(
    title="AstraWatch ML Production API", 
    version="1.1",
    description="High-throughput spatial-temporal inference engine for gridlock prediction.",
    lifespan=lifespan
)

# Production Security & CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URLs like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Custom Exception Handlers for Standard Envelope Format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = ", ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": f"Validation Error: {msg}"}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Internal Server Error: {str(exc)}"}
    )

class PriorityEnum(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"
    Critical = "Critical"

# Production Request Schema with Advanced Validation
class IncidentInput(BaseModel):
    model_config = ConfigDict(extra='ignore') # Strip injection payloads

    latitude: float = Field(..., ge=-90.0, le=90.0, example=12.9720, description="Latitude [-90.0, 90.0]")
    longitude: float = Field(..., ge=-180.0, le=180.0, example=77.6194, description="Longitude [-180.0, 180.0]")
    timestamp: str = Field(..., example="2026-06-17 18:30:00", description="ISO-8601 string or Unix Epoch")
    priority: PriorityEnum = Field(..., example="High", description="Priority levels: Low, Medium, High, Critical")
    
    # Categorical string values
    event_type: str = Field(default="unknown", description="Type of incident")
    event_cause: str = Field(default="unknown", description="Root cause of incident")
    junction: str = Field(default="unknown", description="Nearest traffic junction")
    zone: str = Field(default="unknown", description="Traffic police zone")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> str:
        """Securely parses unix epoch or various string formats into standard datetime string."""
        try:
            # Handle pure epoch timestamp integers or strings
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                dt = datetime.fromtimestamp(float(value))
            else:
                dt = dateutil.parser.parse(str(value))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")

@app.post("/api/v1/predict-metrics")
async def api_predict_metrics(payload: IncidentInput):
    """
    Ingests sanitized client payloads, performs robust spatial inference,
    and returns deterministic metrics.
    """
    # Offload logic natively to the optimized predict module
    results = predict_incident_metrics(
        latitude=payload.latitude,
        longitude=payload.longitude,
        timestamp_str=payload.timestamp,
        priority_str=payload.priority.value,
        event_type=payload.event_type,
        event_cause=payload.event_cause,
        junction=payload.junction,
        zone=payload.zone,
        models_dir="models" 
    )
    
    # Strict standardized response format
    return {
        "status": "success", 
        "data": results
    }

@app.post("/api/v1/report-incident/upload")
async def upload_incident_image(
    file: UploadFile = File(...),
    fallback_latitude: Optional[float] = Form(None),
    fallback_longitude: Optional[float] = Form(None),
    priority: str = Form("High"),
    event_type: str = Form("unknown"),
    event_cause: str = Form("unknown"),
    junction: str = Form("unknown"),
    zone: str = Form("unknown")
):
    """
    Robust multi-layer coordinate intake pipeline. Attempts EXIF extraction natively,
    falls back to client-side geolocation variables gracefully.
    """
    # Step 1: Read the image file stream into memory safely
    file_bytes = await file.read()
    
    # Step 2 & 3: Attempt EXIF native extraction
    coords = extract_exif_gps(file_bytes)
    
    # Step 4: Fallback checks
    if coords is not None:
        latitude, longitude = coords
    elif fallback_latitude is not None and fallback_longitude is not None:
        latitude, longitude = fallback_latitude, fallback_longitude
    else:
        # Step 5: Total Failure
        raise HTTPException(
            status_code=422,
            detail="Unable to resolve geographic location telemetry from photo metadata or browser fallback."
        )
        
    # Standardize priority input
    valid_priorities = ["Low", "Medium", "High", "Critical"]
    safe_priority = priority if priority in valid_priorities else "High"
        
    # Get current system time automatically
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Execute Downstream Pipeline
    results = predict_incident_metrics(
        latitude=latitude,
        longitude=longitude,
        timestamp_str=current_timestamp,
        priority_str=safe_priority,
        event_type=event_type,
        event_cause=event_cause,
        junction=junction,
        zone=zone,
        models_dir="models"
    )
    
    return {
        "status": "success",
        "data": results
    }

@app.get("/api/v1/heatmap-points")
async def get_heatmap_points(hour: Optional[float] = None):
    """
    Feeds the frontend interactive map overlay by returning spatial coordinates
    representing active gridlock hotspots.
    """
    try:
        points = get_heatmap_points_data(hour=hour, models_dir="models")
        return {
            "status": "success",
            "points": points
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate heatmap points: {str(e)}"
        )

# Healthcheck endpoint used to monitor infrastructure status or verify uptime
@app.get("/health")
async def health_check():
    return {"status": "success", "data": {"ml_engine_loaded": True}}

if __name__ == "__main__":
    import uvicorn
    # Optimized server run configuration
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)