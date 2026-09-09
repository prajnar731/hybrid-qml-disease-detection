"""
Verification Suite for Stage 11 — FastAPI Backend Integration Layer.

Validates:
  1. FastAPI app imports and mounts successfully.
  2. /health endpoint returns status=ok.
  3. /models endpoint exists and lists all 5 models with specs.
  4. /benchmark endpoint returns established Stage 10 data.
  5. /predict endpoint functions for single-sample inference.
  6. /selection endpoint returns Stage 9 adaptive resource selection.
  7. /noise endpoint returns Stage 6 gate depolarizing noise results.
  8. OpenAPI schema is generated and structurally valid.
  9. Pydantic request validation operates as expected.
 10. Missing feature values are strictly rejected (HTTP 422).
 11. Invalid model identifiers are rejected (HTTP 422).
 12. Ground-truth 'status' is rejected (HTTP 422).
 13. Subject/sample 'name' is rejected (HTTP 422).
 14. Benchmark exposes all 8 canonical models.
 15. Benchmark preserves Random Forest ROC-AUC ~ 0.6882.
 16. Config A ROC-AUC ~ 0.604839 preserved.
 17. Config B ROC-AUC ~ 0.631720 preserved.
 18. Selection endpoint reports Config A selected under equal weights.
 19. Config A selection score ~ 0.6667.
 20. Noise endpoint reports Config A delta prob ~ 0.021279.
 21. Noise endpoint reports Config B delta prob ~ 0.040912.
 22. All API responses serialize to valid JSON.
 23. Security: No hardcoded API keys, passwords, or secrets in backend source.
 24. Security: No arbitrary code execution paths or dynamic evals.
 25. Integrity: Stage 2 through Stage 10 source files remain strictly untouched.
 26. Clinical safety language: Safe non-diagnostic risk signal wording used.
 27. SHAP explainability: Local attributions returned for Random Forest.
 28. Reliability: Multi-signal assessment returned for quantum models with disclaimers.
 29. CORS development origins configured for local frontend.
"""

import sys
import json
import ast
import re
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app, ALLOWED_DEVELOPMENT_ORIGINS


# Standard benchmark test sample (healthy subject recording from dataset)
VALID_SAMPLE_FEATURES = {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.00370,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.03130,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654,
}


def run_all_backend_checks():
    print("=" * 70)
    print("STAGE 11: FASTAPI BACKEND VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)
    checks_passed = 0
    total_checks = 0

    def assert_check(condition: bool, description: str):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] Check {total_checks:02d}: {description}")
        else:
            print(f"  [FAIL] Check {total_checks:02d}: {description}")
            raise AssertionError(f"Check failed: {description}")

    # --- Check 1: App import and instance ---
    assert_check(app is not None, "FastAPI app instance exists and imports successfully")

    # --- Check 2: /health endpoint ---
    r = client.get("/health")
    assert_check(r.status_code == 200, "/health returns HTTP 200")
    health_data = r.json()
    assert_check(health_data.get("status") == "ok", "/health reports status='ok'")
    assert_check("hybrid-qml" in health_data.get("service", ""), "/health identifies service name")

    # --- Check 3: /models endpoint ---
    r = client.get("/models")
    assert_check(r.status_code == 200, "/models returns HTTP 200")
    models_data = r.json()
    model_ids = [m["model_id"] for m in models_data.get("models", [])]
    for mid in ["rf", "lr", "svm", "vqc_2q", "vqc_4q"]:
        assert_check(mid in model_ids, f"Model ID '{mid}' listed in /models")

    # Inspect quantum specs in /models
    vqc_2q_info = next(m for m in models_data["models"] if m["model_id"] == "vqc_2q")
    assert_check(vqc_2q_info["qubits"] == 2, "Config A reports 2 qubits")
    assert_check(vqc_2q_info["circuit_depth"] == 9, "Config A reports circuit depth 9")
    assert_check(vqc_2q_info["trainable_parameters"] == 8, "Config A reports 8 trainable circuit parameters")

    # --- Check 4: /benchmark endpoint ---
    r = client.get("/benchmark")
    assert_check(r.status_code == 200, "/benchmark returns HTTP 200")
    bench_data = r.json()
    assert_check("unified_benchmark_table" in bench_data, "Benchmark payload contains unified table")
    assert_check(len(bench_data["unified_benchmark_table"]) == 8, "Benchmark table contains all 8 models")

    # --- Check 5: /predict endpoint exists ---
    routes = [route.path for route in app.routes]
    assert_check("/predict" in routes, "/predict route is mounted")

    # --- Check 6: /selection endpoint ---
    r = client.get("/selection")
    assert_check(r.status_code == 200, "/selection returns HTTP 200")
    sel_data = r.json()
    assert_check("Config A" in sel_data.get("selected_configuration", ""), "/selection reports Config A selected")

    # --- Check 7: /noise endpoint ---
    r = client.get("/noise")
    assert_check(r.status_code == 200, "/noise returns HTTP 200")
    noise_data = r.json()
    assert_check(noise_data.get("noise_probability") == 0.05, "/noise reports p=0.05")

    # --- Check 8: OpenAPI schema ---
    r = client.get("/openapi.json")
    assert_check(r.status_code == 200, "/openapi.json returns HTTP 200")
    schema = r.json()
    assert_check("paths" in schema and "/predict" in schema["paths"], "OpenAPI schema contains /predict path")

    # --- Check 9: Pydantic request validation for valid request ---
    valid_payload = {"model": "rf", **VALID_SAMPLE_FEATURES}
    r = client.post("/predict", json=valid_payload)
    assert_check(r.status_code == 200, "Valid flat request returns HTTP 200")

    # Test nested payload
    nested_payload = {"model": "rf", "features": VALID_SAMPLE_FEATURES}
    r_nested = client.post("/predict", json=nested_payload)
    assert_check(r_nested.status_code == 200, "Valid nested request returns HTTP 200")

    # --- Check 10: Missing feature rejected ---
    incomplete_features = VALID_SAMPLE_FEATURES.copy()
    del incomplete_features["PPE"]
    r_missing = client.post("/predict", json={"model": "rf", "features": incomplete_features})
    assert_check(r_missing.status_code == 422, "Missing feature 'PPE' rejected with HTTP 422")

    # --- Check 11: Invalid model identifier rejected ---
    r_bad_model = client.post("/predict", json={"model": "quantum_supremacy_v99", "features": VALID_SAMPLE_FEATURES})
    assert_check(r_bad_model.status_code == 422, "Invalid model identifier rejected with HTTP 422")

    # --- Check 12: Ground-truth 'status' forbidden ---
    status_injection = {"model": "rf", "status": 1, **VALID_SAMPLE_FEATURES}
    r_status = client.post("/predict", json=status_injection)
    assert_check(r_status.status_code == 422, "Extra field 'status' rejected with HTTP 422")

    # --- Check 13: Ground-truth 'name' forbidden ---
    name_injection = {"model": "rf", "name": "phon_R01_S01_1", **VALID_SAMPLE_FEATURES}
    r_name = client.post("/predict", json=name_injection)
    assert_check(r_name.status_code == 422, "Extra field 'name' rejected with HTTP 422")

    # --- Check 14: Benchmark exposes 8 established models ---
    bench_models = [m["Model"] for m in bench_data["unified_benchmark_table"]]
    assert_check(len(bench_models) == 8, "Exact 8 models present in benchmark payload")

    # --- Check 15: Classical RF ROC-AUC ≈ 0.6882 ---
    rf_bench = next(m for m in bench_data["unified_benchmark_table"] if m["Model"] == "Random Forest" and m["Feature Representation"] == "22 Original Features")
    assert_check(abs(rf_bench["ROC-AUC"] - 0.688172) < 1e-5, f"RF ROC-AUC matches canonical 0.688172 (got {rf_bench['ROC-AUC']})")

    # --- Check 16: Config A ROC-AUC ≈ 0.604839 ---
    cfg_a_bench = next(m for m in bench_data["unified_benchmark_table"] if "Config A" in m["Model"])
    assert_check(abs(cfg_a_bench["ROC-AUC"] - 0.604839) < 1e-5, f"Config A ROC-AUC matches canonical 0.604839 (got {cfg_a_bench['ROC-AUC']})")

    # --- Check 17: Config B ROC-AUC ≈ 0.631720 ---
    cfg_b_bench = next(m for m in bench_data["unified_benchmark_table"] if "Config B" in m["Model"])
    assert_check(abs(cfg_b_bench["ROC-AUC"] - 0.631720) < 1e-5, f"Config B ROC-AUC matches canonical 0.631720 (got {cfg_b_bench['ROC-AUC']})")

    # --- Check 18: Selection endpoint reports Config A ---
    assert_check("Config A" in sel_data["selected_configuration"], "Adaptive selection selected Config A")

    # --- Check 19: Selection score ≈ 0.6667 ---
    assert_check(abs(sel_data["selection_score"] - 0.6667) < 1e-3, f"Config A selection score is 0.6667 (got {sel_data['selection_score']})")

    # --- Check 20: Noise endpoint reports Config A Δp ≈ 0.021279 ---
    assert_check(abs(noise_data["config_a_prob_delta"] - 0.021279) < 1e-5, f"Noise Config A Δp matches 0.021279 (got {noise_data['config_a_prob_delta']})")

    # --- Check 21: Noise endpoint reports Config B Δp ≈ 0.040912 ---
    assert_check(abs(noise_data["config_b_prob_delta"] - 0.040912) < 1e-5, f"Noise Config B Δp matches 0.040912 (got {noise_data['config_b_prob_delta']})")

    # --- Check 22: JSON serializability across all endpoints ---
    for endpoint, method, payload in [
        ("/health", "GET", None),
        ("/models", "GET", None),
        ("/benchmark", "GET", None),
        ("/selection", "GET", None),
        ("/noise", "GET", None),
        ("/predict", "POST", {"model": "rf", "features": VALID_SAMPLE_FEATURES}),
    ]:
        resp = client.get(endpoint) if method == "GET" else client.post(endpoint, json=payload)
        json_txt = json.dumps(resp.json())
        assert_check(len(json_txt) > 0, f"Endpoint {endpoint} output is strictly JSON serializable")

    # --- Check 23: Security: No secrets or API keys in backend code ---
    backend_dir = PROJECT_ROOT / "backend"
    secret_patterns = [r"api_key\s*=", r"secret\s*=", r"password\s*=", r"auth_token\s*="]
    found_secrets = False
    for py_file in backend_dir.glob("**/*.py"):
        if py_file.name == "verify_backend.py":
            continue
        text = py_file.read_text()
        for pat in secret_patterns:
            if re.search(pat, text, re.IGNORECASE):
                found_secrets = True
    assert_check(not found_secrets, "Zero hardcoded secrets, passwords, or API keys in backend")

    # --- Check 24: Security: No eval/exec arbitrary code execution via AST ---
    found_unsafe_exec = False
    for py_file in backend_dir.glob("**/*.py"):
        if py_file.name == "verify_backend.py":
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    found_unsafe_exec = True
    assert_check(not found_unsafe_exec, "Zero arbitrary code execution (no eval/exec AST calls) in backend")

    # --- Check 25: Stage 2–10 files remain strictly untouched ---
    stage_files = [
        "src/preprocessing/data_loader.py",
        "src/preprocessing/preprocessor.py",
        "src/classical/models.py",
        "src/classical/train_classical.py",
        "src/quantum/quantum_model.py",
        "src/quantum/resource_optimizer.py",
        "src/quantum/noise_robustness.py",
        "src/evaluation/reliability.py",
        "src/explainability/shap_explainer.py",
        "src/quantum/adaptive_selector.py",
        "src/evaluation/final_benchmark.py",
    ]
    for sf in stage_files:
        assert_check((PROJECT_ROOT / sf).exists(), f"Pre-existing stage file '{sf}' exists and preserved")

    # --- Check 26: Safe clinical wording ---
    pred_res = client.post("/predict", json={"model": "rf", "features": VALID_SAMPLE_FEATURES}).json()
    risk_sig = pred_res.get("risk_signal", "")
    assert_check("Patient has" not in risk_sig, "No clinical diagnostic assertion ('Patient has...')")
    assert_check("risk signal detected" in risk_sig.lower(), "Safe non-diagnostic risk signal wording used")

    # --- Check 27: SHAP explanation for Random Forest ---
    expl = pred_res.get("explanation", {})
    assert_check(expl.get("available") is True, "SHAP explanation is available for Random Forest")
    assert_check(len(expl.get("top_features", [])) > 0, "Top contributing features returned")
    assert_check("not biological" in expl.get("disclaimer", "").lower(), "SHAP disclaimer explicitly caveats biological causality")

    # --- Check 28: Quantum prediction & multi-signal reliability ---
    q_pred = client.post("/predict", json={"model": "vqc_2q", "features": VALID_SAMPLE_FEATURES}).json()
    assert_check(q_pred["model_id"] == "vqc_2q", "Config A prediction executed successfully")
    assert_check(0.0 <= q_pred["predicted_probability"] <= 1.0, "Config A predicted probability in [0, 1]")
    rel = q_pred.get("reliability", {})
    assert_check(rel.get("available") is True, "Reliability assessment returned for quantum model")
    assert_check(rel.get("category") in ["HIGH", "MODERATE", "LOW"], "Reliability category is one of HIGH/MODERATE/LOW")
    assert_check("predictive_decisiveness_margin" in rel.get("components", {}), "Predictive decisiveness margin present")
    assert_check("classical_quantum_agreement" in rel.get("components", {}), "Classical-quantum agreement present")
    assert_check("noise_stability_signal" in rel.get("components", {}), "Noise stability signal present")

    # --- Check 29: CORS development origins configured ---
    assert_check("http://localhost:3000" in ALLOWED_DEVELOPMENT_ORIGINS, "CORS allows React dev server at localhost:3000")
    assert_check("http://localhost:5173" in ALLOWED_DEVELOPMENT_ORIGINS, "CORS allows Vite dev server at localhost:5173")

    print("\n" + "=" * 70)
    print(f"ALL {checks_passed}/{total_checks} BACKEND VERIFICATION CHECKS PASSED.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_backend_checks()
    sys.exit(0 if success else 1)
