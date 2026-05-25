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

<<<<<<< HEAD
from api.config import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    FEATURE_OPTIONS,
    MODELS_DIR,
    get_all_possible_models,
    refresh_model_registry,
)
=======
from api.config import MODEL_REGISTRY, DEFAULT_MODEL, FEATURE_OPTIONS, MODELS_DIR
>>>>>>> origin/dev
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
    RocDataResponse,
    RocCurve,
    RocModelResult,
    RocComparisonResponse,
    ModelComparisonResponse,
    ModelComparisonResult,
<<<<<<< HEAD
    AvailableModelInfo,
    AvailableModelsResponse,
    DatasetConfigResponse,
=======
>>>>>>> origin/dev
)
from api.prediction import (
    predict,
    predict_by_type,
    model_manager,
    get_all_zones,
    predict_by_zone_id,
    predict_all_zones,
)
from api.accuracy_service import (
    evaluate_accuracy,
    evaluate_all_models,
    get_map_data,
    get_roc_data,
    get_all_models_roc_data,
)
from api.chicago_client import ChicagoAPIError
<<<<<<< HEAD
from api.middleware import setup_middleware, setup_request_context_logging
from api.metrics import metrics_router
=======
>>>>>>> origin/dev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Add request context to all log messages
setup_request_context_logging()
logger = logging.getLogger(__name__)

=======
>>>>>>> origin/dev

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

<<<<<<< HEAD
# Add observability middleware (request ID, logging, latency tracking)
setup_middleware(app)

# Include metrics endpoints
app.include_router(metrics_router)

=======
>>>>>>> origin/dev

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


<<<<<<< HEAD
@app.post("/api/reload-model", tags=["Model"])
async def reload_model(model_name: str | None = None):
    """Force reload the model from disk, clearing any cached version.

    Use this after training a new model to ensure the API uses the latest version.
    """
    try:
        model_manager.reload_model(model_name)
        return {
            "status": "success",
            "message": f"Model '{model_manager.get_current_model_name()}' reloaded successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


=======
>>>>>>> origin/dev
@app.post(
    "/api/predict",
    response_model=PredictionResponse
    | ZonePredictionResponse
    | RegressionPredictionResponse
    | HierarchicalPredictionResponse,
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


<<<<<<< HEAD
@app.get("/api/models", response_model=AvailableModelsResponse, tags=["Models"])
async def list_models(include_unavailable: bool = False):
    """List available prediction models with dataset configuration.

    Returns information about all models including which datasets they were
    trained with. Can optionally include unavailable (not-yet-trained) models.

    Args:
        include_unavailable: If True, includes all possible model combinations
                           even if not trained yet (marked as is_available=False).
                           Default False returns only trained models.

    Returns:
        AvailableModelsResponse with list of models and their configurations.
    """
    # Get models based on flag
    if include_unavailable:
        all_models = get_all_possible_models()
    else:
        all_models = MODEL_REGISTRY

    models = []
    available_count = 0

    for key, info in sorted(all_models.items()):
        # Build dataset config response
        datasets = DatasetConfigResponse(
            use_vehicles=info.datasets.use_vehicles,
            use_people=info.datasets.use_people,
            use_weather=info.datasets.use_weather,
            suffix=info.datasets.get_suffix(),
            display_name=info.datasets.get_display_name(),
        )

        models.append(
            AvailableModelInfo(
                key=key,
                name=info.name,
                description=info.description,
                model_type=info.model_type,
                datasets=datasets,
                is_available=info.is_available,
            )
        )

        if info.is_available:
            available_count += 1

    return AvailableModelsResponse(
        models=models,
        available_count=available_count,
        total_count=len(models),
        dataset_combinations=8,
    )


@app.post("/api/models/refresh", tags=["Models"])
async def refresh_models():
    """Refresh the model registry by re-scanning the models directory.

    Call this after training new models to make them available for selection.
    """
    refresh_model_registry()
    return {
        "status": "success",
        "message": f"Model registry refreshed: {len(MODEL_REGISTRY)} models found",
        "models": list(MODEL_REGISTRY.keys()),
    }
=======
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
>>>>>>> origin/dev


# ============================================================================
# Accuracy Evaluation Endpoints
# ============================================================================


def _parse_date(date_str: str | None):
    """Parse a date string (YYYY-MM-DD) to datetime."""
    from datetime import datetime as dt

    if not date_str:
        return None
    try:
        return dt.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


@app.get("/api/accuracy", response_model=AccuracyResponse, tags=["Accuracy"])
async def get_accuracy(
    days: int | None = None,
    max_crashes: int = 10000,
    model: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Evaluate model accuracy on recent real crash data from Chicago.

    Fetches recent crash data from the City of Chicago's open data portal,
    runs predictions, and compares against actual outcomes.

    Args:
        days: Number of days back to fetch data (1, 7, 30, or 90). Used if start_date/end_date not provided.
        max_crashes: Maximum number of crashes to evaluate (default 10000, max 10000)
        model: Model to evaluate (simplified_3class, hierarchical_5class, simplified_zones).
               Defaults to simplified_3class.
        start_date: Start of date range (YYYY-MM-DD format)
        end_date: End of date range (YYYY-MM-DD format)

    Returns:
        AccuracyResponse with metrics and individual predictions
    """
    # Parse date parameters
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    # If date range provided, use it; otherwise validate days
    if parsed_start and parsed_end:
        # Use date range
        pass
    elif days is not None:
        # Validate days parameter
        if days not in [1, 7, 30, 90]:
            days = 7  # Default to 7 days if invalid
    else:
        # Default to 7 days
        days = 7

    # Cap max_crashes to prevent excessive API calls
    max_crashes = min(max(max_crashes, 10), 10000)

    # Validate model parameter
    if model is not None and model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available models: {list(MODEL_REGISTRY.keys())}",
        )

    # Skip regression model for accuracy evaluation
    if model is not None and MODEL_REGISTRY[model].model_type == "regression":
        raise HTTPException(
            status_code=400,
            detail="Regression model is not supported for accuracy evaluation",
        )

    try:
        result = evaluate_accuracy(
            days=days,
            max_crashes=max_crashes,
            model_name=model,
            start_date=parsed_start,
            end_date=parsed_end,
        )

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
            f1_macro=result["metrics"]["f1_macro"],
            f1_micro=result["metrics"]["f1_micro"],
            model_name=result["metrics"].get("model_name"),
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Accuracy evaluation error")
        raise HTTPException(
            status_code=500, detail=f"Accuracy evaluation failed: {str(e)}"
        )


@app.get(
    "/api/accuracy/compare", response_model=ModelComparisonResponse, tags=["Accuracy"]
)
async def get_accuracy_comparison(
    days: int | None = None,
    max_crashes: int = 2000,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Compare accuracy of all available classification models.

    Evaluates all classification models (simplified, hierarchical, zones)
    on the same dataset for fair comparison.

    Args:
        days: Number of days back to fetch data (1, 7, 30, or 90). Used if start_date/end_date not provided.
        max_crashes: Maximum number of crashes per model (default 2000 for faster comparison)
        start_date: Start of date range (YYYY-MM-DD format)
        end_date: End of date range (YYYY-MM-DD format)

    Returns:
        ModelComparisonResponse with metrics for each model
    """
    # Parse date parameters
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    # If date range provided, use it; otherwise validate days
    if parsed_start and parsed_end:
        pass
    elif days is not None:
        if days not in [1, 7, 30, 90]:
            days = 7
    else:
        days = 7

    # Cap max_crashes for comparison (keep lower for performance)
    max_crashes = min(max(max_crashes, 100), 5000)

    try:
        result = evaluate_all_models(
            days=days,
            max_crashes=max_crashes,
            start_date=parsed_start,
            end_date=parsed_end,
        )

        # Convert to response models
        models = {}
        for model_name, model_result in result["models"].items():
            metrics = None
            if model_result["metrics"] is not None:
                metrics = AccuracyMetrics(
                    overall_accuracy=model_result["metrics"]["overall_accuracy"],
                    sample_count=model_result["metrics"]["sample_count"],
                    per_class_metrics={
                        k: ClassMetrics(**v)
                        for k, v in model_result["metrics"]["per_class_metrics"].items()
                    },
                    confusion_matrix=model_result["metrics"]["confusion_matrix"],
                    class_labels=model_result["metrics"]["class_labels"],
                    time_range_days=model_result["metrics"]["time_range_days"],
                    computed_at=model_result["metrics"]["computed_at"],
                    f1_macro=model_result["metrics"]["f1_macro"],
                    f1_micro=model_result["metrics"]["f1_micro"],
                    model_name=model_result["model_name"],
                )

            models[model_name] = ModelComparisonResult(
                model_name=model_result["model_name"],
                display_name=model_result["display_name"],
                model_type=model_result["model_type"],
                metrics=metrics,
                status=model_result["status"],
                error=model_result.get("error"),
            )

        return ModelComparisonResponse(
            models=models,
            time_range_days=result["time_range_days"],
            max_crashes=result["max_crashes"],
            computed_at=result["computed_at"],
        )

    except ChicagoAPIError as e:
        logger.error(f"Chicago API error: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch data from Chicago API: {str(e)}"
        )
    except Exception as e:
        logger.exception("Model comparison error")
        raise HTTPException(
            status_code=500, detail=f"Model comparison failed: {str(e)}"
        )


@app.get("/api/accuracy/map", response_model=MapDataResponse, tags=["Accuracy"])
async def get_accuracy_map_data(
    days: int | None = None,
    max_crashes: int = 200,
    filter: str | None = None,
    model: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Get prediction data formatted for map visualization.

    Returns crash predictions with coordinates for displaying on a map.

    Args:
        days: Number of days back to fetch data. Used if start_date/end_date not provided.
        max_crashes: Maximum number of crashes (default 200 for map performance)
        filter: Optional filter - 'correct', 'incorrect', or None for all
        model: Model to use (defaults to simplified_3class)
        start_date: Start of date range (YYYY-MM-DD format)
        end_date: End of date range (YYYY-MM-DD format)

    Returns:
        MapDataResponse with prediction coordinates and summary counts
    """
    # Parse date parameters
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    # If date range provided, use it; otherwise validate days
    if parsed_start and parsed_end:
        pass
    elif days is not None:
        if days not in [1, 7, 30, 90]:
            days = 7
    else:
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
            model_name=model,
            start_date=parsed_start,
            end_date=parsed_end,
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


# ============================================================================
# ROC Curve Data Endpoint
# ============================================================================


@app.get(
    "/api/models/{model_name}/roc", response_model=RocDataResponse, tags=["Models"]
)
async def get_model_roc_data(model_name: str):
    """Get ROC curve data for a trained model.

    Returns pre-computed ROC curve data (FPR/TPR points and AUC scores)
    generated during model training. Used for frontend ROC visualizations.

    Args:
        model_name: Name of the model (e.g., 'simplified_3class', 'hierarchical_5class')

    Returns:
        RocDataResponse with curve data for each classifier level
    """
    import json

    # Validate model name
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}",
        )

    # Look for ROC data file
    roc_path = MODELS_DIR / model_name / "roc_data.json"

    if not roc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"ROC data not found for model '{model_name}'. "
            "Re-train the model to generate ROC data.",
        )

    try:
        with open(roc_path, "r") as f:
            data = json.load(f)
        return RocDataResponse(**data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in ROC data file: {e}")
        raise HTTPException(
            status_code=500,
            detail="ROC data file is corrupted. Re-train the model.",
        )
    except Exception as e:
        logger.exception("Error reading ROC data")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read ROC data: {str(e)}",
        )


@app.get("/api/accuracy/roc", response_model=RocDataResponse, tags=["Accuracy"])
async def get_accuracy_roc_data(
    days: int | None = None,
    max_crashes: int = 2000,
    model: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Get ROC curve data computed from live accuracy evaluation.

    Runs predictions on recent crash data and computes ROC curves for each
    classification level of the model.

    Args:
        days: Number of days back to fetch data (1, 7, 30, or 90). Used if start_date/end_date not provided.
        max_crashes: Maximum number of crashes to evaluate
        model: Model to evaluate (defaults to simplified_3class)
        start_date: Start of date range (YYYY-MM-DD format)
        end_date: End of date range (YYYY-MM-DD format)

    Returns:
        RocDataResponse with curve data for each classifier level
    """
    # Parse date parameters
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    # If date range provided, use it; otherwise validate days
    if parsed_start and parsed_end:
        pass
    elif days is not None:
        if days not in [1, 7, 30, 90]:
            days = 7
    else:
        days = 7

    # Cap max_crashes
    max_crashes = min(max(max_crashes, 100), 5000)

    # Validate model parameter
    if model is not None and model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available models: {list(MODEL_REGISTRY.keys())}",
        )

    # Skip regression model
    if model is not None and MODEL_REGISTRY[model].model_type == "regression":
        raise HTTPException(
            status_code=400,
            detail="Regression model is not supported for ROC curves",
        )

    try:
        result = get_roc_data(
            days=days,
            max_crashes=max_crashes,
            model_name=model,
            start_date=parsed_start,
            end_date=parsed_end,
        )

        # Convert to response model
        curves = [RocCurve(**c) for c in result["curves"]]

        return RocDataResponse(
            model_name=result["model_name"],
            model_type=result["model_type"],
            curves=curves,
            computed_at=result["computed_at"],
        )

    except ChicagoAPIError as e:
        logger.error(f"Chicago API error: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch data from Chicago API: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ROC data error")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute ROC data: {str(e)}"
        )


@app.get(
    "/api/accuracy/roc/compare", response_model=RocComparisonResponse, tags=["Accuracy"]
)
async def get_roc_comparison(
    days: int | None = None,
    max_crashes: int = 2000,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Get ROC curve data for all classification models.

    Computes ROC curves for each model on the same dataset for comparison.

    Args:
        days: Number of days back to fetch data (1, 7, 30, or 90). Used if start_date/end_date not provided.
        max_crashes: Maximum number of crashes per model
        start_date: Start of date range (YYYY-MM-DD format)
        end_date: End of date range (YYYY-MM-DD format)

    Returns:
        RocComparisonResponse with ROC data for each model
    """
    # Parse date parameters
    parsed_start = _parse_date(start_date)
    parsed_end = _parse_date(end_date)

    # If date range provided, use it; otherwise validate days
    if parsed_start and parsed_end:
        pass
    elif days is not None:
        if days not in [1, 7, 30, 90]:
            days = 7
    else:
        days = 7

    # Cap max_crashes
    max_crashes = min(max(max_crashes, 100), 5000)

    try:
        result = get_all_models_roc_data(
            days=days,
            max_crashes=max_crashes,
            start_date=parsed_start,
            end_date=parsed_end,
        )

        # Convert to response models
        models = {}
        for model_name, model_result in result["models"].items():
            curves = [RocCurve(**c) for c in model_result["curves"]]

            models[model_name] = RocModelResult(
                model_name=model_result["model_name"],
                display_name=model_result["display_name"],
                model_type=model_result["model_type"],
                curves=curves,
                status=model_result["status"],
                error=model_result.get("error"),
            )

        return RocComparisonResponse(
            models=models,
            time_range_days=result["time_range_days"],
            max_crashes=result["max_crashes"],
            computed_at=result["computed_at"],
        )

    except ChicagoAPIError as e:
        logger.error(f"Chicago API error: {e}")
        raise HTTPException(
            status_code=503, detail=f"Failed to fetch data from Chicago API: {str(e)}"
        )
    except Exception as e:
        logger.exception("ROC comparison error")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute ROC comparison: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
