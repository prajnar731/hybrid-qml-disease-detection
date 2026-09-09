"""
FastAPI Main Application — Stage 11 API Layer.

Exposes RESTful endpoints for the Hybrid Quantum Machine Learning Platform:
  1. GET  /health     - Service health verification.
  2. GET  /models     - Catalog of supported classical and quantum models with specs.
  3. GET  /benchmark  - Consolidated Stage 10 benchmarks and performance limits.
  4. POST /predict    - Single-sample inference with SHAP explanations and reliability.
  5. GET  /selection  - Stage 9 multi-objective adaptive quantum resource selection.
  6. GET  /noise      - Stage 6 gate-level depolarizing quantum noise experiment.

Security & Safety:
  - Strict input validation via Pydantic (extra fields like 'status' and 'name' forbidden).
  - Explicit non-diagnostic risk language ('Elevated/Low Parkinson's risk signal detected.').
  - Localhost-only CORS for secure frontend communication.
  - Generic 500 handlers preventing stack trace leakage.
"""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    NoiseResponse,
    AdaptiveSelectionResponse,
)
from backend.services.prediction_service import get_prediction_service


# ------------------------------------------------------------------
# FastAPI Application Configuration
# ------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Quantum Machine Learning Platform API",
    description=(
        "REST API for early disease detection research (Parkinson's vocal acoustic biomarkers). "
        "Integrates Classical ML baselines (Stage 3), Variational Quantum Classifiers (Stages 4-6), "
        "Reliability-Aware Prediction (Stage 7), SHAP Interpretability (Stage 8), Adaptive "
        "Quantum Resource Optimization (Stage 9), and Unified Benchmarking (Stage 10)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ------------------------------------------------------------------
# CORS Configuration (Restricted to Localhost Development Origins)
# ------------------------------------------------------------------

ALLOWED_DEVELOPMENT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_DEVELOPMENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Secure Exception Handlers
# ------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Returns structured 422 errors without revealing internal path details."""
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", [])])
        msg = err.get("msg", "Invalid input")
        err_type = err.get("type", "validation_error")
        errors.append({"field": field, "message": msg, "type": err_type})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Request validation failed", "errors": errors},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles business logic and data processing errors cleanly."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors and returns safe generic response without stack traces."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error_type": type(exc).__name__,
        },
    )


# ------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Returns service availability and platform status.",
)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="hybrid-qml-disease-detection")


@app.get(
    "/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List Available Models",
    description="Returns specifications, qubit allocations, circuit depths, and disclaimers for all available models.",
)
def list_models() -> ModelsListResponse:
    service = get_prediction_service()
    return service.get_models_list()


@app.get(
    "/benchmark",
    tags=["Evaluation"],
    summary="Consolidated Stage 10 Benchmarks",
    description="Returns the established Stage 10 consolidated benchmark data across all 8 models and quantum resource tables.",
)
def get_benchmark() -> Dict[str, Any]:
    service = get_prediction_service()
    return service.get_benchmark_payload()


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Single-Sample Disease Risk Screening",
    description=(
        "Performs single-sample inference using the requested model. Rejects missing features "
        "and ground-truth labels ('status', 'name'). For Random Forest, provides local SHAP "
        "attributions. For quantum models, provides evidence-based reliability signals."
    ),
)
def predict_sample(payload: PredictRequest) -> PredictResponse:
    service = get_prediction_service()
    features_dict = payload.features.to_feature_dict()
    try:
        response = service.predict(features_dict=features_dict, model_id=payload.model)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {type(e).__name__}",
        )


@app.get(
    "/selection",
    response_model=AdaptiveSelectionResponse,
    tags=["Quantum Resources"],
    summary="Adaptive Quantum Resource Selection",
    description="Returns the established Stage 9 multi-objective resource selection outcome, scores, and Pareto frontier.",
)
def get_adaptive_selection() -> AdaptiveSelectionResponse:
    service = get_prediction_service()
    return service.get_adaptive_selection_data()


@app.get(
    "/noise",
    response_model=NoiseResponse,
    tags=["Quantum Robustness"],
    summary="Quantum Noise Robustness Evaluation",
    description="Returns established Stage 6 gate-level depolarizing quantum noise results (p=0.05 on default.mixed).",
)
def get_noise_robustness() -> NoiseResponse:
    service = get_prediction_service()
    return service.get_noise_data()
