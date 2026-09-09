"""
Pydantic Data Schemas and Validation Models — Stage 11 API Layer.

Defines strict request/response contracts for the Hybrid QML Platform API:
  - BiomedicalVoiceFeatures: Exact 22 acoustic features (with aliases and forbidden extras).
  - PredictRequest: Model selection and feature payload (rejects ground-truth 'status', 'name').
  - PredictResponse: Prediction, probability, safe risk signal, SHAP explanation, reliability signals.
  - Benchmark, Models, Noise, Selection, and Health schemas.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


# Canonical order of the 22 biomedical voice features in the UCI Oxford dataset
CANONICAL_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


class ModelIdentifier(str, Enum):
    """Supported prediction models and quantum configurations."""
    RANDOM_FOREST = "rf"
    LOGISTIC_REGRESSION = "lr"
    SVM = "svm"
    VQC_CONFIG_A = "vqc_2q"
    VQC_CONFIG_B = "vqc_4q"


class BiomedicalVoiceFeatures(BaseModel):
    """
    Exact 22 biomedical voice measurements required for Parkinson's disease risk screening.
    Rejects any unlisted keys (such as 'name', 'status', 'subject_id') strictly via extra='forbid'.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fo_hz: float = Field(..., alias="MDVP:Fo(Hz)", description="Average vocal fundamental frequency (Hz)")
    fhi_hz: float = Field(..., alias="MDVP:Fhi(Hz)", description="Maximum vocal fundamental frequency (Hz)")
    flo_hz: float = Field(..., alias="MDVP:Flo(Hz)", description="Minimum vocal fundamental frequency (Hz)")
    jitter_percent: float = Field(..., alias="MDVP:Jitter(%)", description="MDVP local jitter percentage (%)")
    jitter_abs: float = Field(..., alias="MDVP:Jitter(Abs)", description="MDVP absolute jitter in microseconds")
    rap: float = Field(..., alias="MDVP:RAP", description="MDVP Relative Amplitude Perturbation")
    ppq: float = Field(..., alias="MDVP:PPQ", description="MDVP Five-point Period Perturbation Quotient")
    jitter_ddp: float = Field(..., alias="Jitter:DDP", description="Average absolute difference of differences between jitter cycles")
    shimmer: float = Field(..., alias="MDVP:Shimmer", description="MDVP local shimmer")
    shimmer_db: float = Field(..., alias="MDVP:Shimmer(dB)", description="MDVP local shimmer in decibels (dB)")
    shimmer_apq3: float = Field(..., alias="Shimmer:APQ3", description="Three-point Amplitude Perturbation Quotient")
    shimmer_apq5: float = Field(..., alias="Shimmer:APQ5", description="Five-point Amplitude Perturbation Quotient")
    apq: float = Field(..., alias="MDVP:APQ", description="MDVP 11-point Amplitude Perturbation Quotient")
    shimmer_dda: float = Field(..., alias="Shimmer:DDA", description="Average absolute differences between consecutive amplitude differences")
    nhr: float = Field(..., alias="NHR", description="Noise-to-Harmonics Ratio")
    hnr: float = Field(..., alias="HNR", description="Harmonics-to-Noise Ratio")
    rpde: float = Field(..., alias="RPDE", description="Recurrence Period Density Entropy")
    dfa: float = Field(..., alias="DFA", description="Detrended Fluctuation Analysis")
    spread1: float = Field(..., alias="spread1", description="Nonlinear fundamental frequency variation parameter 1")
    spread2: float = Field(..., alias="spread2", description="Nonlinear fundamental frequency variation parameter 2")
    d2: float = Field(..., alias="D2", description="Correlation dimension")
    ppe: float = Field(..., alias="PPE", description="Pitch Period Entropy")

    def to_feature_dict(self) -> Dict[str, float]:
        """Returns ordered feature dictionary keyed by official UCI dataset column names."""
        return {
            "MDVP:Fo(Hz)": float(self.fo_hz),
            "MDVP:Fhi(Hz)": float(self.fhi_hz),
            "MDVP:Flo(Hz)": float(self.flo_hz),
            "MDVP:Jitter(%)": float(self.jitter_percent),
            "MDVP:Jitter(Abs)": float(self.jitter_abs),
            "MDVP:RAP": float(self.rap),
            "MDVP:PPQ": float(self.ppq),
            "Jitter:DDP": float(self.jitter_ddp),
            "MDVP:Shimmer": float(self.shimmer),
            "MDVP:Shimmer(dB)": float(self.shimmer_db),
            "Shimmer:APQ3": float(self.shimmer_apq3),
            "Shimmer:APQ5": float(self.shimmer_apq5),
            "MDVP:APQ": float(self.apq),
            "Shimmer:DDA": float(self.shimmer_dda),
            "NHR": float(self.nhr),
            "HNR": float(self.hnr),
            "RPDE": float(self.rpde),
            "DFA": float(self.dfa),
            "spread1": float(self.spread1),
            "spread2": float(self.spread2),
            "D2": float(self.d2),
            "PPE": float(self.ppe),
        }


class PredictRequest(BaseModel):
    """
    Prediction request supporting either a nested 'features' object or flat top-level features.
    Explicitly forbids 'name', 'status', 'subject_id', or any other unlisted parameters.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: ModelIdentifier = Field(
        default=ModelIdentifier.RANDOM_FOREST,
        description="Classifier identifier: 'rf', 'lr', 'svm', 'vqc_2q', 'vqc_4q'"
    )
    features: Optional[BiomedicalVoiceFeatures] = Field(
        default=None,
        description="Biomedical voice features nested under 'features'"
    )

    # Optional flat fields matching the 22 features
    fo_hz: Optional[float] = Field(None, alias="MDVP:Fo(Hz)")
    fhi_hz: Optional[float] = Field(None, alias="MDVP:Fhi(Hz)")
    flo_hz: Optional[float] = Field(None, alias="MDVP:Flo(Hz)")
    jitter_percent: Optional[float] = Field(None, alias="MDVP:Jitter(%)")
    jitter_abs: Optional[float] = Field(None, alias="MDVP:Jitter(Abs)")
    rap: Optional[float] = Field(None, alias="MDVP:RAP")
    ppq: Optional[float] = Field(None, alias="MDVP:PPQ")
    jitter_ddp: Optional[float] = Field(None, alias="Jitter:DDP")
    shimmer: Optional[float] = Field(None, alias="MDVP:Shimmer")
    shimmer_db: Optional[float] = Field(None, alias="MDVP:Shimmer(dB)")
    shimmer_apq3: Optional[float] = Field(None, alias="Shimmer:APQ3")
    shimmer_apq5: Optional[float] = Field(None, alias="Shimmer:APQ5")
    apq: Optional[float] = Field(None, alias="MDVP:APQ")
    shimmer_dda: Optional[float] = Field(None, alias="Shimmer:DDA")
    nhr: Optional[float] = Field(None, alias="NHR")
    hnr: Optional[float] = Field(None, alias="HNR")
    rpde: Optional[float] = Field(None, alias="RPDE")
    dfa: Optional[float] = Field(None, alias="DFA")
    spread1: Optional[float] = Field(None, alias="spread1")
    spread2: Optional[float] = Field(None, alias="spread2")
    d2: Optional[float] = Field(None, alias="D2")
    ppe: Optional[float] = Field(None, alias="PPE")

    @model_validator(mode="after")
    def assemble_features(self) -> "PredictRequest":
        """Consolidates features whether provided as a nested dict or flat fields."""
        if self.features is not None:
            return self

        # Check if all 22 flat fields are present
        flat_fields = [
            self.fo_hz, self.fhi_hz, self.flo_hz, self.jitter_percent, self.jitter_abs,
            self.rap, self.ppq, self.jitter_ddp, self.shimmer, self.shimmer_db,
            self.shimmer_apq3, self.shimmer_apq5, self.apq, self.shimmer_dda,
            self.nhr, self.hnr, self.rpde, self.dfa, self.spread1, self.spread2,
            self.d2, self.ppe
        ]
        if any(f is None for f in flat_fields):
            missing = []
            for name, val in [
                ("MDVP:Fo(Hz)", self.fo_hz), ("MDVP:Fhi(Hz)", self.fhi_hz), ("MDVP:Flo(Hz)", self.flo_hz),
                ("MDVP:Jitter(%)", self.jitter_percent), ("MDVP:Jitter(Abs)", self.jitter_abs),
                ("MDVP:RAP", self.rap), ("MDVP:PPQ", self.ppq), ("Jitter:DDP", self.jitter_ddp),
                ("MDVP:Shimmer", self.shimmer), ("MDVP:Shimmer(dB)", self.shimmer_db),
                ("Shimmer:APQ3", self.shimmer_apq3), ("Shimmer:APQ5", self.shimmer_apq5),
                ("MDVP:APQ", self.apq), ("Shimmer:DDA", self.shimmer_dda),
                ("NHR", self.nhr), ("HNR", self.hnr), ("RPDE", self.rpde),
                ("DFA", self.dfa), ("spread1", self.spread1), ("spread2", self.spread2),
                ("D2", self.d2), ("PPE", self.ppe)
            ]:
                if val is None:
                    missing.append(name)
            raise ValueError(f"Missing required biomedical features: {missing}")

        self.features = BiomedicalVoiceFeatures(
            fo_hz=self.fo_hz,
            fhi_hz=self.fhi_hz,
            flo_hz=self.flo_hz,
            jitter_percent=self.jitter_percent,
            jitter_abs=self.jitter_abs,
            rap=self.rap,
            ppq=self.ppq,
            jitter_ddp=self.jitter_ddp,
            shimmer=self.shimmer,
            shimmer_db=self.shimmer_db,
            shimmer_apq3=self.shimmer_apq3,
            shimmer_apq5=self.shimmer_apq5,
            apq=self.apq,
            shimmer_dda=self.shimmer_dda,
            nhr=self.nhr,
            hnr=self.hnr,
            rpde=self.rpde,
            dfa=self.dfa,
            spread1=self.spread1,
            spread2=self.spread2,
            d2=self.d2,
            ppe=self.ppe,
        )
        return self


# ------------------------------------------------------------------
# Explainability & Reliability Sub-Schemas
# ------------------------------------------------------------------

class SHAPFeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    scaled_feature_value: float
    direction: str  # 'increases_risk' | 'decreases_risk'
    interpretation: str


class ExplanationResponse(BaseModel):
    available: bool
    method: Optional[str] = None
    base_value: Optional[float] = None
    top_features: Optional[List[SHAPFeatureContribution]] = None
    disclaimer: Optional[str] = None


class ReliabilityComponents(BaseModel):
    predictive_decisiveness_margin: float
    classical_quantum_agreement: float
    noise_stability_signal: float


class ReliabilityResponse(BaseModel):
    available: bool
    status: str
    composite_score: Optional[float] = None
    category: Optional[str] = None  # HIGH | MODERATE | LOW
    components: Optional[ReliabilityComponents] = None
    disclaimer: Optional[str] = None
    cohort_calibration_note: Optional[str] = None


# ------------------------------------------------------------------
# Primary Endpoint Response Schemas
# ------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Structured response for /predict with careful safety language."""
    model_id: str
    model_name: str
    feature_representation: str
    predicted_probability: float
    predicted_class: int
    risk_signal: str  # "Elevated Parkinson's risk signal detected." or "Low Parkinson's risk signal detected."
    decision_threshold: float = 0.5
    explanation: Optional[ExplanationResponse] = None
    reliability: Optional[ReliabilityResponse] = None
    disclaimer: str = (
        "Experimental research prototype for vocal acoustic biomarker processing. "
        "NOT a certified medical diagnostic device. Do not use for clinical treatment decisions."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "hybrid-qml-disease-detection"


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    model_type: str  # 'classical' | 'hybrid_quantum'
    feature_representation: str
    n_features: int
    qubits: Optional[int] = None
    circuit_depth: Optional[int] = None
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    training_time_sec: Optional[float] = None
    description: str
    clinical_disclaimer: str = "Experimental model, not clinically validated for patient diagnosis."


class ModelsListResponse(BaseModel):
    count: int
    models: List[ModelInfo]
    platform_disclaimer: str = (
        "All models are research prototypes trained on the UCI Oxford Parkinson's dataset. "
        "None are certified medical diagnostic instruments."
    )


class NoiseResponse(BaseModel):
    noise_model: str
    noise_channel: str
    noise_probability: float = 0.05
    backend_device: str
    config_a_prob_delta: float
    config_b_prob_delta: float
    interpretation: str
    disclaimer: str


class AdaptiveSelectionResponse(BaseModel):
    selected_configuration: str
    selection_score: float
    selection_weights: Dict[str, float]
    pareto_frontier: List[str]
    selection_rationale: str
    evaluated_configurations: List[Dict[str, Any]]
