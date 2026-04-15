"""FastAPI application for crash severity prediction.

Run with:
    uvicorn api.main:app --reload --port 8000

Or:
    python -m api.main
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import MODEL_REGISTRY, DEFAULT_MODEL, FEATURE_OPTIONS
from api.models import (
    PredictionRequest,
    PredictionResponse,
    FeatureOptionsResponse,
    ModelInfoResponse,
    HealthResponse,
    ZonePredictionResponse,
    RegressionPredictionResponse,
    HierarchicalPredictionResponse,
    ModelType,
    ZoneInfo,
    ZonesResponse,
    ZonePredictionByIdRequest,
    AllZonesPredictionResponse,
)
from api.prediction import predict, predict_by_type, model_manager, get_all_zones, predict_by_zone_id, predict_all_zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    logger.info("Starting API server...")
    try:
        model_manager.load_model(DEFAULT_MODEL)
        logger.info(f"Default model '{DEFAULT_MODEL}' loaded successfully")
    except ValueError as e:
        logger.warning(f"Could not load default model: {e}")
        logger.warning("Model will be loaded on first prediction request")
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Crash Severity Prediction API",
    description="API for predicting crash severity based on various input features",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Crash Severity Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_model_loaded(),
        model_name=model_manager.get_current_model_name(),
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse | ZonePredictionResponse | RegressionPredictionResponse | HierarchicalPredictionResponse,
    tags=["Prediction"],
)
async def predict_severity(request: PredictionRequest):
    """Predict crash severity or count based on input features.

    Takes crash parameters and returns predictions based on the selected model type:
    - simplified: 3-class severity (NO_INJURY, MINOR, SEVERE)
    - hierarchical: 5-class severity (same output format for now)
    - zones: Zone-aware 3-class severity with zone information
    - regression: Predicted crash count
    """
    try:
        return predict_by_type(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/zones", response_model=ZonesResponse, tags=["Zones"])
async def list_zones():
    """Get all geographic zones with their centroids.

    Returns zone IDs and center coordinates for map visualization.
    Zones are K-means clusters of historical crash locations.
    """
    try:
        zones = get_all_zones()
        return ZonesResponse(zones=zones, total_zones=len(zones))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error fetching zones")
        raise HTTPException(status_code=500, detail=f"Failed to fetch zones: {str(e)}")


@app.post(
    "/api/predict/zone/{zone_id}",
    response_model=ZonePredictionResponse,
    tags=["Zones"],
)
async def predict_for_zone(zone_id: int, request: ZonePredictionByIdRequest):
    """Predict crash severity for a specific zone by ID.

    Does not require latitude/longitude - uses the zone's centroid.
    """
    if zone_id != request.zone_id:
        raise HTTPException(
            status_code=400,
            detail=f"Zone ID mismatch: path={zone_id}, body={request.zone_id}",
        )
    try:
        return predict_by_zone_id(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Zone prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/api/predict/zones/all",
    response_model=AllZonesPredictionResponse,
    tags=["Zones"],
)
async def predict_all_zones_endpoint(request: ZonePredictionByIdRequest):
    """Predict crash severity for all zones at once.

    Returns predictions for all zones using the same input features.
    Useful for visualizing severity across the entire city.
    """
    try:
        predictions = predict_all_zones(request)
        return AllZonesPredictionResponse(
            predictions=predictions,
            total_zones=len(predictions),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("All-zones prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/features", response_model=FeatureOptionsResponse, tags=["Features"])
async def get_feature_options():
    """Get available options for categorical features.

    Returns the allowed values for dropdown selection in the frontend.
    """
    return FeatureOptionsResponse(**FEATURE_OPTIONS)


@app.get("/api/models", response_model=list[ModelInfoResponse], tags=["Models"])
async def list_models():
    """List available prediction models.

    Returns information about all models in the registry.
    This endpoint supports future model selection functionality.
    """
    return [
        ModelInfoResponse(
            name=info.name,
            description=info.description,
            model_type=info.model_type,
        )
        for info in MODEL_REGISTRY.values()
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
