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
    AccuracyResponse,
    AccuracyMetrics,
    ClassMetrics,
    PredictionWithActual,
    MapPrediction,
    MapDataResponse,
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


# ============================================================================
# Accuracy Evaluation Endpoints
# ============================================================================


@app.get("/api/accuracy", response_model=AccuracyResponse, tags=["Accuracy"])
async def get_accuracy(
    days: int = 7,
    max_crashes: int = 500,
):
    """Evaluate model accuracy on recent real crash data from Chicago.

    Fetches recent crash data from the City of Chicago's open data portal,
    runs predictions, and compares against actual outcomes.

    Args:
        days: Number of days back to fetch data (1, 7, 30, or 90)
        max_crashes: Maximum number of crashes to evaluate (default 500)

    Returns:
        AccuracyResponse with metrics and individual predictions
    """
    # Validate days parameter
    if days not in [1, 7, 30, 90]:
        days = 7  # Default to 7 days if invalid

    # Cap max_crashes to prevent excessive API calls
    max_crashes = min(max(max_crashes, 10), 2000)

    try:
        result = evaluate_accuracy(days=days, max_crashes=max_crashes)

        # Convert to response models
        metrics = AccuracyMetrics(
            overall_accuracy=result["metrics"]["overall_accuracy"],
            sample_count=result["metrics"]["sample_count"],
            per_class_metrics={
                k: ClassMetrics(**v)
                for k, v in result["metrics"]["per_class_metrics"].items()
            },
            confusion_matrix=result["metrics"]["confusion_matrix"],
            class_labels=result["metrics"]["class_labels"],
            time_range_days=result["metrics"]["time_range_days"],
            computed_at=result["metrics"]["computed_at"],
        )

        predictions = [PredictionWithActual(**p) for p in result["predictions"]]

        return AccuracyResponse(metrics=metrics, predictions=predictions)

    except ChicagoAPIError as e:
        logger.error(f"Chicago API error: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch data from Chicago API: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Accuracy evaluation error")
        raise HTTPException(
            status_code=500, detail=f"Accuracy evaluation failed: {str(e)}"
        )


@app.get("/api/accuracy/map", response_model=MapDataResponse, tags=["Accuracy"])
async def get_accuracy_map_data(
    days: int = 7,
    max_crashes: int = 200,
    filter: str | None = None,
):
    """Get prediction data formatted for map visualization.

    Returns crash predictions with coordinates for displaying on a map.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes (default 200 for map performance)
        filter: Optional filter - 'correct', 'incorrect', or None for all

    Returns:
        MapDataResponse with prediction coordinates and summary counts
    """
    # Validate days parameter
    if days not in [1, 7, 30, 90]:
        days = 7

    # Cap max_crashes for map performance
    max_crashes = min(max(max_crashes, 10), 500)

    # Parse filter parameter
    filter_correct = None
    if filter == "correct":
        filter_correct = True
    elif filter == "incorrect":
        filter_correct = False

    try:
        map_data = get_map_data(
            days=days,
            max_crashes=max_crashes,
            filter_correct=filter_correct,
        )

        # Convert to response model
        predictions = [MapPrediction(**p) for p in map_data]

        correct_count = sum(1 for p in predictions if p.is_correct)
        incorrect_count = len(predictions) - correct_count

        return MapDataResponse(
            predictions=predictions,
            total_count=len(predictions),
            correct_count=correct_count,
            incorrect_count=incorrect_count,
        )

    except ChicagoAPIError as e:
        logger.error(f"Chicago API error: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch data from Chicago API: {str(e)}"
        )
    except Exception as e:
        logger.exception("Map data error")
        raise HTTPException(status_code=500, detail=f"Failed to get map data: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
