"""
Prediction and ML/QML Integration Service — Stage 11.

Integrates the existing Stage 2 through Stage 10 pipeline modules:
  - Stage 2: Subject-aware data preprocessor & StandardScaler (fitted strictly on training split).
  - Stage 3: Classical ML baselines (Random Forest, Logistic Regression, SVM).
  - Stage 4/5: Variational Quantum Classifiers (Config A: 2Q, Config B: 4Q) initialized with
               exact canonical trained parameters from experimental checkpoints.
  - Stage 6: Gate-level depolarizing quantum noise circuit evaluation (p=0.05).
  - Stage 7: Evidence-based per-sample reliability signals (Decisiveness, Agreement, Noise Stability).
  - Stage 8: Single-sample SHAP TreeExplainer local feature attributions.
  - Stage 9: Adaptive quantum resource selection metadata.
  - Stage 10: Canonical consolidated benchmark table export.

Does NOT modify any existing Stage 2–10 files.
Does NOT fabricate predictions or reliability scores.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
from pennylane import numpy as pnp

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.preprocessor import ParkinsonsPreprocessor
from src.classical.models import get_classical_models
from src.quantum.quantum_model import VariationalQuantumClassifier
from src.quantum.noise_robustness import build_noisy_qnode
from src.evaluation.reliability import classify_reliability_category
from src.evaluation.final_benchmark import assemble_complete_benchmark, CANONICAL_BENCHMARK_DATA

from backend.schemas import (
    ModelIdentifier,
    CANONICAL_FEATURE_NAMES,
    PredictResponse,
    ExplanationResponse,
    SHAPFeatureContribution,
    ReliabilityResponse,
    ReliabilityComponents,
    ModelInfo,
    ModelsListResponse,
    NoiseResponse,
    AdaptiveSelectionResponse,
)


# ------------------------------------------------------------------
# Canonical Quantum Model Parameters (from Stages 4, 5, 6, 9)
# ------------------------------------------------------------------

CONFIG_A_CANONICAL_WEIGHTS = [
    -1.45496989, 0.00606745, -0.77064475, 0.15230299,
    -0.02341534, -0.02341370, 1.12983177, 0.07674347, 1.25384474
]

CONFIG_B_CANONICAL_WEIGHTS = [
    -1.41337301, -0.01614795, -0.94244566, -0.96196011,
    -1.05672941, 1.03395968, -0.09830913, 1.13540415,
    -0.04694744, 0.05425600, -1.79064645, -0.04657298,
    1.80446077, -0.19132802, -1.54963751, -0.05622875, 1.27292461
]


class PredictionService:
    """
    Central backend service managing model initialization, caching, and inference.
    Fitted once strictly on the training set to prevent data leakage.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._is_initialized = False

        # Preprocessors
        self.prep_classical: Optional[ParkinsonsPreprocessor] = None
        self.prep_pca2: Optional[ParkinsonsPreprocessor] = None
        self.prep_pca4: Optional[ParkinsonsPreprocessor] = None

        # Classical models
        self.rf_model = None
        self.lr_model = None
        self.svm_model = None
        self.tree_explainer: Optional[shap.TreeExplainer] = None

        # Quantum models & noisy circuits
        self.vqc_2q: Optional[VariationalQuantumClassifier] = None
        self.vqc_4q: Optional[VariationalQuantumClassifier] = None
        self.noisy_circuit_2q = None
        self.noisy_circuit_4q = None

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Fits preprocessors and initializes models strictly on training split."""
        if self._is_initialized:
            return

        # 1. Load raw data and split reproducibly
        df = load_raw_data()

        # Classical Preprocessor (22 Original Features)
        self.prep_classical = ParkinsonsPreprocessor(use_pca=False, random_state=self.random_state)
        X_train, _, y_train, _, _, _ = self.prep_classical.split_data(
            df, test_size=0.20, random_state=self.random_state
        )
        X_train_scaled = self.prep_classical.fit_transform_train(X_train)

        # 2. Fit Classical Models
        models = get_classical_models(random_state=self.random_state)
        self.rf_model = models["Random Forest"]
        self.rf_model.fit(X_train_scaled, y_train)

        self.lr_model = models["Logistic Regression"]
        self.lr_model.fit(X_train_scaled, y_train)

        self.svm_model = models["Support Vector Machine"]
        self.svm_model.fit(X_train_scaled, y_train)

        # 3. Fit SHAP TreeExplainer for Random Forest
        self.tree_explainer = shap.TreeExplainer(self.rf_model)

        # 4. Fit 2-Component PCA Preprocessor & Initialize Config A VQC
        self.prep_pca2 = ParkinsonsPreprocessor(
            use_pca=True, n_components=2, random_state=self.random_state
        )
        X_train_pca2 = self.prep_pca2.fit_transform_train(X_train)

        self.vqc_2q = VariationalQuantumClassifier(
            n_qubits=2, n_layers=2, random_state=self.random_state
        )
        self.vqc_2q.params = pnp.array(CONFIG_A_CANONICAL_WEIGHTS, requires_grad=False)
        self.vqc_2q.min_val = X_train_pca2.min(axis=0)
        self.vqc_2q.max_val = X_train_pca2.max(axis=0)
        self.vqc_2q.is_fitted = True

        self.noisy_circuit_2q, _ = build_noisy_qnode(n_qubits=2, n_layers=2)

        # 5. Fit 4-Component PCA Preprocessor & Initialize Config B VQC
        self.prep_pca4 = ParkinsonsPreprocessor(
            use_pca=True, n_components=4, random_state=self.random_state
        )
        X_train_pca4 = self.prep_pca4.fit_transform_train(X_train)

        self.vqc_4q = VariationalQuantumClassifier(
            n_qubits=4, n_layers=2, random_state=self.random_state
        )
        self.vqc_4q.params = pnp.array(CONFIG_B_CANONICAL_WEIGHTS, requires_grad=False)
        self.vqc_4q.min_val = X_train_pca4.min(axis=0)
        self.vqc_4q.max_val = X_train_pca4.max(axis=0)
        self.vqc_4q.is_fitted = True

        self.noisy_circuit_4q, _ = build_noisy_qnode(n_qubits=4, n_layers=2)

        self._is_initialized = True

    def predict(
        self,
        features_dict: Dict[str, float],
        model_id: ModelIdentifier = ModelIdentifier.RANDOM_FOREST
    ) -> PredictResponse:
        """
        Executes genuine single-sample inference for the specified model configuration:
          - Applies established StandardScaler (and PCA for quantum).
          - Computes real continuous probability.
          - Discretizes prediction using established 0.5 threshold.
          - Provides local SHAP feature attributions for Random Forest.
          - Computes genuine multi-signal reliability assessment for quantum models.
        """
        # Convert dictionary to ordered DataFrame row
        sample_df = pd.DataFrame([[features_dict[col] for col in CANONICAL_FEATURE_NAMES]], columns=CANONICAL_FEATURE_NAMES)

        # Dispatch based on model_id
        if model_id == ModelIdentifier.RANDOM_FOREST:
            return self._predict_random_forest(sample_df)
        elif model_id == ModelIdentifier.LOGISTIC_REGRESSION:
            return self._predict_classical_linear(sample_df, model=self.lr_model, model_id="lr", name="Logistic Regression")
        elif model_id == ModelIdentifier.SVM:
            return self._predict_classical_linear(sample_df, model=self.svm_model, model_id="svm", name="Support Vector Machine")
        elif model_id == ModelIdentifier.VQC_CONFIG_A:
            return self._predict_quantum(sample_df, n_qubits=2, model_id="vqc_2q", name="Hybrid VQC (Config A — 2 Qubits)")
        elif model_id == ModelIdentifier.VQC_CONFIG_B:
            return self._predict_quantum(sample_df, n_qubits=4, model_id="vqc_4q", name="Hybrid VQC (Config B — 4 Qubits)")
        else:
            raise ValueError(f"Unsupported model identifier: {model_id}")

    def _predict_random_forest(self, sample_df: pd.DataFrame) -> PredictResponse:
        """Runs genuine single-sample inference and SHAP explanation for Random Forest."""
        X_scaled = self.prep_classical.transform(sample_df)
        probs = self.rf_model.predict_proba(X_scaled)[0]
        prob_pos = float(probs[1])
        pred_class = 1 if prob_pos >= 0.5 else 0

        risk_signal = (
            "Elevated Parkinson's risk signal detected."
            if pred_class == 1
            else "Low Parkinson's risk signal detected."
        )

        # Compute SHAP local attributions
        raw_shap = self.tree_explainer.shap_values(X_scaled)
        if isinstance(raw_shap, list):
            shap_pos = raw_shap[1][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_pos = raw_shap[0, :, 1]
        else:
            shap_pos = raw_shap[0]

        if hasattr(self.tree_explainer.expected_value, "__len__") and len(self.tree_explainer.expected_value) > 1:
            base_val = float(self.tree_explainer.expected_value[1])
        else:
            base_val = float(self.tree_explainer.expected_value)

        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(shap_pos))[::-1]
        top_features = []
        for idx in sorted_indices[:5]:
            feat_name = CANONICAL_FEATURE_NAMES[idx]
            val_shap = float(shap_pos[idx])
            raw_val = float(sample_df.iloc[0][feat_name])
            sc_val = float(X_scaled[0, idx])
            direction = "increases_risk" if val_shap > 0 else "decreases_risk"
            interp = (
                f"Pushed prediction toward elevated risk by {abs(val_shap):.4f}"
                if val_shap > 0
                else f"Pushed prediction toward lower risk by {abs(val_shap):.4f}"
            )
            top_features.append(
                SHAPFeatureContribution(
                    feature=feat_name,
                    shap_value=round(val_shap, 4),
                    feature_value=round(raw_val, 4),
                    scaled_feature_value=round(sc_val, 4),
                    direction=direction,
                    interpretation=interp,
                )
            )

        explanation = ExplanationResponse(
            available=True,
            method="SHAP TreeExplainer (Exact Tree Kernel)",
            base_value=round(base_val, 4),
            top_features=top_features,
            disclaimer="SHAP feature attributions describe internal model decision influence, NOT biological or clinical causality.",
        )

        reliability = ReliabilityResponse(
            available=False,
            status="not_applicable_for_classical_model",
            disclaimer="Multi-signal quantum reliability assessment is designated for hybrid quantum models.",
        )

        return PredictResponse(
            model_id="rf",
            model_name="Random Forest",
            feature_representation="22 Original Features",
            predicted_probability=round(prob_pos, 4),
            predicted_class=pred_class,
            risk_signal=risk_signal,
            decision_threshold=0.5,
            explanation=explanation,
            reliability=reliability,
        )

    def _predict_classical_linear(
        self, sample_df: pd.DataFrame, model: Any, model_id: str, name: str
    ) -> PredictResponse:
        """Inference for Logistic Regression or SVM."""
        X_scaled = self.prep_classical.transform(sample_df)
        probs = model.predict_proba(X_scaled)[0]
        prob_pos = float(probs[1])
        pred_class = 1 if prob_pos >= 0.5 else 0

        risk_signal = (
            "Elevated Parkinson's risk signal detected."
            if pred_class == 1
            else "Low Parkinson's risk signal detected."
        )

        return PredictResponse(
            model_id=model_id,
            model_name=name,
            feature_representation="22 Original Features",
            predicted_probability=round(prob_pos, 4),
            predicted_class=pred_class,
            risk_signal=risk_signal,
            decision_threshold=0.5,
            explanation=ExplanationResponse(
                available=False,
                method=None,
                disclaimer="SHAP explanations are currently configured for Random Forest.",
            ),
            reliability=ReliabilityResponse(
                available=False,
                status="not_applicable_for_classical_model",
            ),
        )

    def _predict_quantum(
        self, sample_df: pd.DataFrame, n_qubits: int, model_id: str, name: str
    ) -> PredictResponse:
        """Runs genuine PennyLane circuit inference and multi-signal reliability assessment."""
        preprocessor = self.prep_pca2 if n_qubits == 2 else self.prep_pca4
        vqc = self.vqc_2q if n_qubits == 2 else self.vqc_4q
        noisy_circuit = self.noisy_circuit_2q if n_qubits == 2 else self.noisy_circuit_4q

        # Transform through PCA and normalize to [-pi, pi]
        X_pca = preprocessor.transform(sample_df)
        X_norm = vqc.transform_features(X_pca)

        # 1. Genuine PennyLane QNode circuit evaluation (default.qubit)
        weights_arr = vqc.params[:vqc.n_circuit_params].reshape((vqc.n_layers, vqc.n_qubits, 2))
        bias = float(vqc.params[vqc.n_circuit_params])
        ideal_expval = float(vqc.qnode(weights_arr, X_norm[0]))
        prob_q = float(1.0 / (1.0 + np.exp(-(ideal_expval + bias))))
        pred_class = 1 if prob_q >= 0.5 else 0

        risk_signal = (
            "Elevated Parkinson's risk signal detected."
            if pred_class == 1
            else "Low Parkinson's risk signal detected."
        )

        # 2. Genuine Multi-Signal Reliability Evaluation (Stage 7)
        # Signal 1: Predictive Decisiveness (Distance from 0.5 boundary)
        s_dec = float(2.0 * abs(prob_q - 0.5))

        # Signal 2: Classical-Quantum Agreement (Paired Random Forest reference)
        X_scaled_rf = self.prep_classical.transform(sample_df)
        prob_rf = float(self.rf_model.predict_proba(X_scaled_rf)[0, 1])
        s_agr = float(1.0 - abs(prob_q - prob_rf))

        # Signal 3: Gate Depolarizing Noise Stability (p=0.05 on default.mixed)
        noisy_weights = np.array(weights_arr)
        noisy_feat = np.array(X_norm[0])
        noisy_expval = float(noisy_circuit(noisy_weights, noisy_feat, 0.05))
        prob_noisy = float(1.0 / (1.0 + np.exp(-(noisy_expval + bias))))
        s_noise = float(1.0 - abs(prob_q - prob_noisy))

        # Composite score: Equal 1/3 weighting
        composite_score = float((1.0 / 3.0) * (s_dec + s_agr + s_noise))
        rel_category = classify_reliability_category(composite_score)

        reliability = ReliabilityResponse(
            available=True,
            status="available",
            composite_score=round(composite_score, 4),
            category=rel_category,
            components=ReliabilityComponents(
                predictive_decisiveness_margin=round(s_dec, 4),
                classical_quantum_agreement=round(s_agr, 4),
                noise_stability_signal=round(s_noise, 4),
            ),
            disclaimer=(
                "Experimental research reliability score; NOT a probability of correctness, "
                "NOT clinical confidence, and NOT diagnostic certainty."
            ),
            cohort_calibration_note=(
                "Model-level calibration metrics (Brier Score and ECE) are cohort-level evaluations "
                "and are not defined as scalar per-patient scores."
            ),
        )

        explanation = ExplanationResponse(
            available=False,
            method=None,
            disclaimer=(
                "SHAP local attributions are configured for the 22 original features via Random Forest; "
                "quantum classifier operates over PCA-reduced quantum state rotations."
            ),
        )

        return PredictResponse(
            model_id=model_id,
            model_name=name,
            feature_representation=f"{n_qubits} PCA Components",
            predicted_probability=round(prob_q, 4),
            predicted_class=pred_class,
            risk_signal=risk_signal,
            decision_threshold=0.5,
            explanation=explanation,
            reliability=reliability,
        )

    def get_models_list(self) -> ModelsListResponse:
        """Returns catalog of all classical and quantum models available."""
        models = [
            ModelInfo(
                model_id="rf",
                model_name="Random Forest",
                model_type="classical",
                feature_representation="22 Original Features",
                n_features=22,
                qubits=None,
                circuit_depth=None,
                trainable_parameters=None,
                total_parameters=None,
                training_time_sec=0.12,
                description="Ensemble classifier with highest continuous discriminability (ROC-AUC 0.6882).",
            ),
            ModelInfo(
                model_id="lr",
                model_name="Logistic Regression",
                model_type="classical",
                feature_representation="22 Original Features",
                n_features=22,
                qubits=None,
                circuit_depth=None,
                trainable_parameters=None,
                total_parameters=None,
                training_time_sec=0.05,
                description="Standard linear baseline for vocal biomarker classification.",
            ),
            ModelInfo(
                model_id="svm",
                model_name="Support Vector Machine",
                model_type="classical",
                feature_representation="22 Original Features",
                n_features=22,
                qubits=None,
                circuit_depth=None,
                trainable_parameters=None,
                total_parameters=None,
                training_time_sec=0.08,
                description="RBF kernel support vector classifier with Platt scaling probabilities.",
            ),
            ModelInfo(
                model_id="vqc_2q",
                model_name="Hybrid VQC (Config A)",
                model_type="hybrid_quantum",
                feature_representation="2 PCA Components",
                n_features=2,
                qubits=2,
                circuit_depth=9,
                trainable_parameters=8,
                total_parameters=9,
                training_time_sec=30.271,
                description="Resource-efficient 2-qubit VQC selected under multi-objective criteria (Score: 0.6667).",
            ),
            ModelInfo(
                model_id="vqc_4q",
                model_name="Hybrid VQC (Config B)",
                model_type="hybrid_quantum",
                feature_representation="4 PCA Components",
                n_features=4,
                qubits=4,
                circuit_depth=13,
                trainable_parameters=16,
                total_parameters=17,
                training_time_sec=57.000,
                description="4-qubit VQC configuration maximizing quantum discriminability (ROC-AUC 0.6317).",
            ),
        ]
        return ModelsListResponse(count=len(models), models=models)

    def get_benchmark_payload(self) -> Dict[str, Any]:
        """Returns unified Stage 10 benchmark data."""
        return assemble_complete_benchmark()

    def get_noise_data(self) -> NoiseResponse:
        """Returns Stage 6 quantum noise experiment summary."""
        cfg_a = CANONICAL_BENCHMARK_DATA["quantum_configurations"]["Config A (2 Qubits / 2 PCA)"]
        cfg_b = CANONICAL_BENCHMARK_DATA["quantum_configurations"]["Config B (4 Qubits / 4 PCA)"]
        return NoiseResponse(
            noise_model="DepolarizingChannel (gate-level after single-qubit rotations and entangling CNOTs)",
            noise_channel="qml.DepolarizingChannel",
            noise_probability=0.05,
            backend_device="default.mixed",
            config_a_prob_delta=cfg_a["noise_prob_delta"],
            config_b_prob_delta=cfg_b["noise_prob_delta"],
            interpretation=(
                "Depolarizing noise contracts quantum expectation values toward zero. Config A demonstrates "
                "48% less probability shift than Config B due to lower circuit depth (9 vs 13) and fewer gates. "
                "The models are NOT universally noise-robust; discrete predictions remained stable only because "
                "probabilities remained on the positive side of the 0.5 threshold under tested noise regimes."
            ),
            disclaimer="Simulated gate-level noise on density matrix backend; physical NISQ execution may encounter additional environmental decoherence.",
        )

    def get_adaptive_selection_data(self) -> AdaptiveSelectionResponse:
        """Returns Stage 9 adaptive quantum resource selection outcome."""
        summary = CANONICAL_BENCHMARK_DATA["adaptive_selection_summary"]
        configs = list(CANONICAL_BENCHMARK_DATA["quantum_configurations"].values())
        return AdaptiveSelectionResponse(
            selected_configuration=summary["selected_configuration"],
            selection_score=0.6667,
            selection_weights=summary["selection_weights"],
            pareto_frontier=summary["pareto_frontier"],
            selection_rationale=summary["selection_rationale"],
            evaluated_configurations=configs,
        )


# Global service singleton
_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Provides lazy-initialized singleton instance of PredictionService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance
