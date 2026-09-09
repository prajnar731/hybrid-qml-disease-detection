"""
Adaptive Quantum Resource Selection — Stage 9.

Implements the platform's core innovation: Adaptive Quantum Resource Optimization.
Selects among evaluated quantum configurations by quantitatively balancing:
  1. Predictive Performance: Normalized ROC-AUC (rewarded)
  2. Resource Cost: Normalized combination of qubits, parameters, depth, and training time (penalized)
  3. Physical Quantum Noise Robustness: Normalized resistance to gate depolarizing noise (rewarded)

Among the evaluated configurations, Config A identifies the lower-resource
operating point under the predefined selection criteria.

Reuses the exact measured values from Stage 5 (Resource Evaluation) and Stage 6 (Noise Robustness).
Does NOT use Stage 7 reliability scores or Stage 8 SHAP attributions (avoids circular optimization).
Provides multi-objective weighted selection scoring and empirical Pareto-efficiency analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Measured Benchmark Results from Stages 5 & 6
# ------------------------------------------------------------------

STAGE_5_6_BENCHMARKS = {
    "Config A (2 Qubits / 2 PCA)": {
        "config_label": "Config A (2 Qubits / 2 PCA)",
        "n_qubits": 2,
        "n_pca_features": 2,
        "n_circuit_params": 8,
        "n_total_params": 9,
        "circuit_depth": 9,
        "training_time_sec": 30.271,
        "roc_auc": 0.604839,
        "accuracy": 0.720930,
        "mean_noise_prob_delta": 0.021279,
        "mean_expval_shift": 0.011188,
    },
    "Config B (4 Qubits / 4 PCA)": {
        "config_label": "Config B (4 Qubits / 4 PCA)",
        "n_qubits": 4,
        "n_pca_features": 4,
        "n_circuit_params": 16,
        "n_total_params": 17,
        "circuit_depth": 13,
        "training_time_sec": 57.000,
        "roc_auc": 0.631720,
        "accuracy": 0.720930,
        "mean_noise_prob_delta": 0.040912,
        "mean_expval_shift": 0.068952,
    },
}


# ------------------------------------------------------------------
# Normalization Helper Functions
# ------------------------------------------------------------------

def min_max_normalize(val: float, min_val: float, max_val: float) -> float:
    """Normalizes scalar val into [0, 1]. If min == max, returns 0.0."""
    if abs(max_val - min_val) < 1e-9:
        return 0.0
    return float((val - min_val) / (max_val - min_val))


# ------------------------------------------------------------------
# Adaptive Quantum Resource Selector Class
# ------------------------------------------------------------------

class AdaptiveQuantumResourceSelector:
    """
    Multi-objective selector that evaluates and selects quantum configurations
    based on predictive performance, quantum resource efficiency, and noise robustness.
    """

    def __init__(
        self,
        weight_performance: float = 1.0 / 3.0,
        weight_resource: float = 1.0 / 3.0,
        weight_noise: float = 1.0 / 3.0,
    ):
        weights_sum = weight_performance + weight_resource + weight_noise
        assert abs(weights_sum - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weights_sum}"

        self.w_perf = float(weight_performance)
        self.w_res = float(weight_resource)
        self.w_noise = float(weight_noise)

    def evaluate_configurations(
        self,
        configs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates candidate configurations:
          1. Normalizes predictive performance (ROC-AUC).
          2. Normalizes resource costs (qubits, parameters, depth, training time).
          3. Normalizes noise sensitivity (mean probability shift under depolarizing noise).
          4. Computes composite selection score.
          5. Performs Pareto dominance analysis.
          6. Selects highest-scoring configuration.
        """
        if configs is None:
            configs = list(STAGE_5_6_BENCHMARKS.values())

        if len(configs) < 2:
            raise ValueError("At least two configurations are required for relative selection.")

        # Extract range boundaries for normalization
        aucs = [c["roc_auc"] for c in configs]
        qubits = [c["n_qubits"] for c in configs]
        params = [c["n_total_params"] for c in configs]
        depths = [c["circuit_depth"] for c in configs]
        times = [c["training_time_sec"] for c in configs]
        noise_deltas = [c["mean_noise_prob_delta"] for c in configs]

        min_auc, max_auc = min(aucs), max(aucs)
        min_q, max_q = min(qubits), max(qubits)
        min_p, max_p = min(params), max(params)
        min_d, max_d = min(depths), max(depths)
        min_t, max_t = min(times), max(times)
        min_nd, max_nd = min(noise_deltas), max(noise_deltas)

        evaluated_list = []
        for c in configs:
            # 1. Normalized Predictive Performance (higher is better)
            norm_perf = min_max_normalize(c["roc_auc"], min_auc, max_auc)

            # 2. Normalized Resource Cost (qubits, params, depth, time)
            norm_q = min_max_normalize(c["n_qubits"], min_q, max_q)
            norm_p = min_max_normalize(c["n_total_params"], min_p, max_p)
            norm_d = min_max_normalize(c["circuit_depth"], min_d, max_d)
            norm_t = min_max_normalize(c["training_time_sec"], min_t, max_t)

            # Equal-weighted aggregate resource cost in [0, 1]
            resource_cost = 0.25 * (norm_q + norm_p + norm_d + norm_t)
            # Resource Efficiency: 1 - cost (higher is better)
            resource_efficiency = 1.0 - resource_cost

            # 3. Normalized Noise Robustness (lower delta is better)
            # Noise sensitivity: 0 = least sensitive (best), 1 = most sensitive (worst)
            noise_sensitivity = min_max_normalize(c["mean_noise_prob_delta"], min_nd, max_nd)
            # Noise robustness: 1 - sensitivity (higher is better)
            noise_robustness = 1.0 - noise_sensitivity

            # 4. Multi-Objective Selection Score
            selection_score = (
                self.w_perf * norm_perf +
                self.w_res * resource_efficiency +
                self.w_noise * noise_robustness
            )

            evaluated_list.append({
                "config_label": c["config_label"],
                "n_qubits": c["n_qubits"],
                "n_pca_features": c["n_pca_features"],
                "n_total_params": c["n_total_params"],
                "circuit_depth": c["circuit_depth"],
                "training_time_sec": c["training_time_sec"],
                "roc_auc": c["roc_auc"],
                "mean_noise_prob_delta": c["mean_noise_prob_delta"],
                # Normalized components
                "normalized_performance": norm_perf,
                "normalized_resource_cost": resource_cost,
                "normalized_resource_efficiency": resource_efficiency,
                "normalized_noise_robustness": noise_robustness,
                "selection_score": selection_score,
                "raw_config": c,
            })

        # 5. Determine Selected Configuration (highest score)
        scores = [e["selection_score"] for e in evaluated_list]
        max_score_idx = int(np.argmax(scores))
        for idx, e in enumerate(evaluated_list):
            e["is_selected"] = bool(idx == max_score_idx)

        selected_config = evaluated_list[max_score_idx]

        # 6. Pareto Efficiency Analysis
        # A config is dominated if another has >= perf, >= noise_rob, <= all resource costs, with at least one strictly better
        pareto_results = self._analyze_pareto_efficiency(evaluated_list)

        # 7. Build Formatted Summary Table
        table_rows = []
        for e in evaluated_list:
            table_rows.append({
                "Configuration": e["config_label"],
                "Qubits": e["n_qubits"],
                "Depth": e["circuit_depth"],
                "Params": e["n_total_params"],
                "Time (s)": e["training_time_sec"],
                "ROC-AUC": e["roc_auc"],
                "Noise Δp": e["mean_noise_prob_delta"],
                "Norm Perf": round(e["normalized_performance"], 4),
                "Norm Res Eff": round(e["normalized_resource_efficiency"], 4),
                "Norm Noise Rob": round(e["normalized_noise_robustness"], 4),
                "Selection Score": round(e["selection_score"], 4),
                "Selected": "★ YES" if e["is_selected"] else "NO",
                "Pareto Optimal": "YES" if pareto_results[e["config_label"]]["is_pareto_optimal"] else "NO",
            })
        summary_df = pd.DataFrame(table_rows)

        return {
            "evaluated_configurations": evaluated_list,
            "selected_configuration": selected_config,
            "selection_weights": {
                "weight_performance": self.w_perf,
                "weight_resource": self.w_res,
                "weight_noise": self.w_noise,
            },
            "pareto_analysis": pareto_results,
            "summary_df": summary_df,
            "normalization_bounds": {
                "auc_range": (min_auc, max_auc),
                "qubit_range": (min_q, max_q),
                "param_range": (min_p, max_p),
                "depth_range": (min_d, max_d),
                "time_range": (min_t, max_t),
                "noise_delta_range": (min_nd, max_nd),
            }
        }

    def _analyze_pareto_efficiency(
        self,
        evaluated_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determines empirical Pareto efficiency.
        Objective space:
          - Maximize: ROC-AUC
          - Maximize: Noise Robustness (1 - delta_p)
          - Minimize: Qubits, Parameters, Depth, Training Time
        """
        results = {}
        for i, a in enumerate(evaluated_list):
            is_dominated = False
            dominated_by = []
            for j, b in enumerate(evaluated_list):
                if i == j:
                    continue
                # Check if b dominates a:
                # b must be >= in all maximization objectives and <= in all minimization objectives,
                # and strictly better in at least one objective.
                perf_better_or_equal = b["roc_auc"] >= a["roc_auc"]
                noise_better_or_equal = b["mean_noise_prob_delta"] <= a["mean_noise_prob_delta"]
                q_better_or_equal = b["n_qubits"] <= a["n_qubits"]
                p_better_or_equal = b["n_total_params"] <= a["n_total_params"]
                d_better_or_equal = b["circuit_depth"] <= a["circuit_depth"]
                t_better_or_equal = b["training_time_sec"] <= a["training_time_sec"]

                all_better_or_equal = (
                    perf_better_or_equal and
                    noise_better_or_equal and
                    q_better_or_equal and
                    p_better_or_equal and
                    d_better_or_equal and
                    t_better_or_equal
                )

                strictly_better = (
                    (b["roc_auc"] > a["roc_auc"]) or
                    (b["mean_noise_prob_delta"] < a["mean_noise_prob_delta"]) or
                    (b["n_qubits"] < a["n_qubits"]) or
                    (b["n_total_params"] < a["n_total_params"]) or
                    (b["circuit_depth"] < a["circuit_depth"]) or
                    (b["training_time_sec"] < a["training_time_sec"])
                )

                if all_better_or_equal and strictly_better:
                    is_dominated = True
                    dominated_by.append(b["config_label"])

            results[a["config_label"]] = {
                "is_pareto_optimal": not is_dominated,
                "is_dominated": is_dominated,
                "dominated_by": dominated_by,
            }
        return results


def run_adaptive_selection(
    weight_performance: float = 1.0 / 3.0,
    weight_resource: float = 1.0 / 3.0,
    weight_noise: float = 1.0 / 3.0,
) -> Dict[str, Any]:
    """Convenience function to run the full adaptive resource selection analysis."""
    selector = AdaptiveQuantumResourceSelector(
        weight_performance=weight_performance,
        weight_resource=weight_resource,
        weight_noise=weight_noise,
    )
    return selector.evaluate_configurations()
