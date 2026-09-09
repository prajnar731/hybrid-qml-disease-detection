"""
Full System Integration & End-to-End Verification Suite — Stage 13.

Verifies complete cross-stage integration:
  Frontend (React) -> FastAPI Backend -> Preprocessor -> Models (Classical & Quantum)
  -> Reliability / SHAP / Noise Analysis -> Response Serialization -> Frontend Compatibility.

Validates:
  1. Backend integration endpoints: /health, /models, /benchmark, /selection, /noise, /docs.
  2. Prediction end-to-end flow for all 5 models (LR, RF, SVM, VQC 2Q, VQC 4Q).
  3. Input validation & negative testing (missing, extra, non-numeric, status, name, malformed JSON).
  4. RF + SHAP explanation flow (additivity, directional contributions, causality disclaimers).
  5. Quantum multi-signal reliability flow (Decisiveness, Agreement, Noise Stability, Categories).
  6. Frontend/Backend contract alignment (API base URL, CORS, model identifiers, feature names).
  7. Real process socket smoke test (live HTTP server requests + Vite production bundle).
  8. Cross-stage consistency (Stages 2 through 12).
  9. Security and data integrity (zero leakage, zero hardcoded experimental metrics, zero secrets).
"""

import sys
import json
import ast
import re
import time
import threading
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import uvicorn
from fastapi.testclient import TestClient

from backend.main import app as fastapi_app, ALLOWED_DEVELOPMENT_ORIGINS
from backend.schemas import CANONICAL_FEATURE_NAMES


# Canonical test sample from UCI dataset test cohort (healthy patient phonation)
CANONICAL_TEST_SAMPLE = {
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


def run_full_system_verification():
    print("=" * 70)
    print("STAGE 13: FULL SYSTEM INTEGRATION & END-TO-END VERIFICATION")
    print("=" * 70)

    client = TestClient(fastapi_app)
    checks_passed = 0
    total_checks = 0

    section_status = {
        "Backend endpoints": True,
        "Prediction flow": True,
        "Input validation": True,
        "RF + SHAP flow": True,
        "Quantum reliability flow": True,
        "Frontend API integration": True,
        "Production build": True,
        "Cross-stage consistency": True,
        "Security/hardcoding checks": True,
    }

    def record_check(section: str, cond: bool, description: str):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if cond:
            checks_passed += 1
            print(f"  [PASS] Check {total_checks:02d} [{section}]: {description}")
        else:
            section_status[section] = False
            print(f"  [FAIL] Check {total_checks:02d} [{section}]: {description}")
            raise AssertionError(f"Check failed in {section}: {description}")

    # ==================================================================
    # 1. Backend Integration Tests
    # ==================================================================
    print("\n--- 1. Backend Integration Tests ---")
    sec = "Backend endpoints"

    # GET /health
    r = client.get("/health")
    record_check(sec, r.status_code == 200, "GET /health returns HTTP 200")
    record_check(sec, r.json().get("status") == "ok", "/health reports status='ok'")
    record_check(sec, "hybrid-qml" in r.json().get("service", ""), "/health identifies service")

    # GET /models
    r = client.get("/models")
    record_check(sec, r.status_code == 200, "GET /models returns HTTP 200")
    models_list = r.json().get("models", [])
    record_check(sec, len(models_list) == 5, "/models lists exactly 5 models")
    m_ids = [m["model_id"] for m in models_list]
    for expected_id in ["rf", "lr", "svm", "vqc_2q", "vqc_4q"]:
        record_check(sec, expected_id in m_ids, f"Model '{expected_id}' present in /models")

    # GET /benchmark
    r = client.get("/benchmark")
    record_check(sec, r.status_code == 200, "GET /benchmark returns HTTP 200")
    bench = r.json()
    record_check(sec, "unified_benchmark_table" in bench, "Benchmark payload contains unified table")
    record_check(sec, len(bench["unified_benchmark_table"]) == 8, "Benchmark table contains all 8 models")
    record_check(sec, "metric_leaders" in bench, "Benchmark payload contains metric leaders")
    record_check(sec, "limitations" in bench, "Benchmark payload contains explicit limitations")

    # GET /selection
    r = client.get("/selection")
    record_check(sec, r.status_code == 200, "GET /selection returns HTTP 200")
    sel = r.json()
    record_check(sec, "Config A" in sel["selected_configuration"], "/selection identifies Config A as selected")
    record_check(sec, abs(sel["selection_score"] - 0.6667) < 1e-3, "Selection score is 0.6667")
    record_check(sec, len(sel["pareto_frontier"]) == 2, "Pareto frontier contains both configurations")

    # GET /noise
    r = client.get("/noise")
    record_check(sec, r.status_code == 200, "GET /noise returns HTTP 200")
    noise = r.json()
    record_check(sec, noise["noise_probability"] == 0.05, "/noise reports primary p=0.05")
    record_check(sec, abs(noise["config_a_prob_delta"] - 0.021279) < 1e-5, "Config A prob delta matches 0.021279")
    record_check(sec, abs(noise["config_b_prob_delta"] - 0.040912) < 1e-5, "Config B prob delta matches 0.040912")

    # GET /docs & /openapi.json
    r_docs = client.get("/docs")
    record_check(sec, r_docs.status_code == 200, "GET /docs returns HTTP 200")
    r_openapi = client.get("/openapi.json")
    record_check(sec, r_openapi.status_code == 200 and "/predict" in r_openapi.json()["paths"], "GET /openapi.json valid")

    # ==================================================================
    # 2. Prediction End-to-End Tests
    # ==================================================================
    print("\n--- 2. Prediction End-to-End Tests ---")
    sec_pred = "Prediction flow"

    models_to_test = [
        ("lr", "Logistic Regression"),
        ("rf", "Random Forest"),
        ("svm", "Support Vector Machine"),
        ("vqc_2q", "Hybrid VQC (Config A — 2 Qubits)"),
        ("vqc_4q", "Hybrid VQC (Config B — 4 Qubits)"),
    ]

    for m_id, m_name in models_to_test:
        payload = {"model": m_id, "features": CANONICAL_TEST_SAMPLE}
        r_pred = client.post("/predict", json=payload)
        record_check(sec_pred, r_pred.status_code == 200, f"POST /predict for '{m_id}' returns HTTP 200")
        data = r_pred.json()
        record_check(sec_pred, data["model_id"] == m_id, f"Response model_id matches '{m_id}'")
        record_check(sec_pred, 0.0 <= data["predicted_probability"] <= 1.0, f"Predicted probability in [0, 1] for '{m_id}'")
        record_check(sec_pred, data["predicted_class"] in (0, 1), f"Predicted class is binary for '{m_id}'")
        record_check(sec_pred, "risk signal detected" in data["risk_signal"].lower(), f"Safe non-diagnostic risk wording for '{m_id}'")
        record_check(sec_pred, "Patient has" not in data["risk_signal"], f"No direct disease assertion for '{m_id}'")

    # ==================================================================
    # 3. Input Validation Tests (Negative Testing)
    # ==================================================================
    print("\n--- 3. Input Validation Tests ---")
    sec_val = "Input validation"

    # Missing required feature
    incomplete_sample = CANONICAL_TEST_SAMPLE.copy()
    del incomplete_sample["PPE"]
    r_missing = client.post("/predict", json={"model": "rf", "features": incomplete_sample})
    record_check(sec_val, r_missing.status_code == 422, "Missing feature 'PPE' rejected with HTTP 422")

    # Extra unsupported feature
    extra_sample = CANONICAL_TEST_SAMPLE.copy()
    extra_sample["unsupported_feature_xyz"] = 123.45
    r_extra = client.post("/predict", json={"model": "rf", "features": extra_sample})
    record_check(sec_val, r_extra.status_code == 422, "Extra unsupported feature rejected with HTTP 422")

    # Non-numeric feature
    bad_type_sample = CANONICAL_TEST_SAMPLE.copy()
    bad_type_sample["HNR"] = "not_a_number"
    r_bad_type = client.post("/predict", json={"model": "rf", "features": bad_type_sample})
    record_check(sec_val, r_bad_type.status_code == 422, "Non-numeric feature rejected with HTTP 422")

    # Invalid model identifier
    r_bad_model = client.post("/predict", json={"model": "quantum_supremacy_v99", "features": CANONICAL_TEST_SAMPLE})
    record_check(sec_val, r_bad_model.status_code == 422, "Invalid model identifier rejected with HTTP 422")

    # Ground-truth 'status' field injection
    r_status = client.post("/predict", json={"model": "rf", "status": 1, **CANONICAL_TEST_SAMPLE})
    record_check(sec_val, r_status.status_code == 422, "Ground-truth 'status' field rejected with HTTP 422")

    # Subject identifier 'name' field injection
    r_name = client.post("/predict", json={"model": "rf", "name": "phon_R01_S01_1", **CANONICAL_TEST_SAMPLE})
    record_check(sec_val, r_name.status_code == 422, "Subject identifier 'name' rejected with HTTP 422")

    # Malformed JSON payload
    r_malformed = client.post("/predict", content="not a json string", headers={"Content-Type": "application/json"})
    record_check(sec_val, r_malformed.status_code in (400, 422), "Malformed JSON returns 4xx error")

    # ==================================================================
    # 4. RF + SHAP Explainability Tests
    # ==================================================================
    print("\n--- 4. RF + SHAP Explainability Tests ---")
    sec_shap = "RF + SHAP flow"

    r_rf = client.post("/predict", json={"model": "rf", "features": CANONICAL_TEST_SAMPLE}).json()
    expl = r_rf.get("explanation", {})
    record_check(sec_shap, expl.get("available") is True, "SHAP explanation available for Random Forest")
    record_check(sec_shap, expl.get("method") is not None, "SHAP method reported")
    record_check(sec_shap, isinstance(expl.get("base_value"), float), "SHAP base_value is a valid float")
    top_feats = expl.get("top_features", [])
    record_check(sec_shap, len(top_feats) > 0, "Top SHAP feature contributions returned")

    # Check structure of top features
    first_feat = top_feats[0]
    record_check(sec_shap, "feature" in first_feat, "Feature name in contribution item")
    record_check(sec_shap, "shap_value" in first_feat, "SHAP attribution value present")
    record_check(sec_shap, first_feat["direction"] in ("increases_risk", "decreases_risk"), "Valid direction in attribution item")
    record_check(sec_shap, "not biological" in expl.get("disclaimer", "").lower(), "Causality disclaimer caveats biological causality")

    # Verify other models report SHAP unavailable cleanly
    r_lr = client.post("/predict", json={"model": "lr", "features": CANONICAL_TEST_SAMPLE}).json()
    record_check(sec_shap, r_lr.get("explanation", {}).get("available") is False, "SHAP gracefully reported unavailable for linear model")

    # ==================================================================
    # 5. Quantum Multi-Signal Reliability Tests
    # ==================================================================
    print("\n--- 5. Quantum Multi-Signal Reliability Tests ---")
    sec_rel = "Quantum reliability flow"

    for q_id in ["vqc_2q", "vqc_4q"]:
        r_q = client.post("/predict", json={"model": q_id, "features": CANONICAL_TEST_SAMPLE}).json()
        rel = r_q.get("reliability", {})
        record_check(sec_rel, rel.get("available") is True, f"Reliability available for '{q_id}'")
        record_check(sec_rel, rel.get("category") in ("HIGH", "MODERATE", "LOW"), f"Valid category for '{q_id}'")
        record_check(sec_rel, 0.0 <= rel.get("composite_score", -1) <= 1.0, f"Composite score in [0, 1] for '{q_id}'")

        comps = rel.get("components", {})
        record_check(sec_rel, "predictive_decisiveness_margin" in comps, f"Decisiveness margin present for '{q_id}'")
        record_check(sec_rel, "classical_quantum_agreement" in comps, f"Classical-quantum agreement present for '{q_id}'")
        record_check(sec_rel, "noise_stability_signal" in comps, f"Noise stability signal present for '{q_id}'")

        disclaimer = rel.get("disclaimer", "")
        record_check(sec_rel, "experimental research reliability" in disclaimer.lower(), f"Research reliability disclaimer present for '{q_id}'")

    # Check classical model reports reliability not applicable
    r_rf = client.post("/predict", json={"model": "rf", "features": CANONICAL_TEST_SAMPLE}).json()
    record_check(sec_rel, r_rf.get("reliability", {}).get("available") is False, "Reliability reported not applicable for classical model")

    # ==================================================================
    # 6. Frontend / Backend Contract Alignment
    # ==================================================================
    print("\n--- 6. Frontend / Backend Contract Alignment ---")
    sec_fe = "Frontend API integration"

    frontend_dir = PROJECT_ROOT / "frontend"
    api_js = (frontend_dir / "src" / "api.js").read_text()

    # Check base URL
    record_check(sec_fe, "http://127.0.0.1:8000" in api_js, "Frontend api.js configures backend URL at 127.0.0.1:8000")

    # Check CORS configuration in FastAPI
    record_check(sec_fe, "http://localhost:3000" in ALLOWED_DEVELOPMENT_ORIGINS, "CORS allows React dev server at localhost:3000")
    record_check(sec_fe, "http://localhost:5173" in ALLOWED_DEVELOPMENT_ORIGINS, "CORS allows Vite dev server at localhost:5173")

    # Check frontend components do NOT hardcode benchmark or noise numbers
    components_dir = frontend_dir / "src" / "components"
    prohibited_hardcodes = ["0.688172", "0.604839", "0.631720", "0.021279", "0.040912"]
    for comp in components_dir.glob("*.jsx"):
        text = comp.read_text()
        for val in prohibited_hardcodes:
            record_check(sec_fe, val not in text, f"Component '{comp.name}' does not hardcode '{val}'")

    # Check all 22 feature names exist in CANONICAL_FEATURE_NAMES
    record_check(sec_fe, len(CANONICAL_FEATURE_NAMES) == 22, "22 canonical features defined in schema")
    for feat in CANONICAL_FEATURE_NAMES:
        record_check(sec_fe, feat in CANONICAL_TEST_SAMPLE, f"Feature '{feat}' in test payload")

    # ==================================================================
    # 7. Real Process Smoke Test (Live Socket Server + Vite Build)
    # ==================================================================
    print("\n--- 7. Real Process Smoke Test ---")
    sec_smoke = "Production build"

    # Check Vite production build assets exist
    dist_index = frontend_dir / "dist" / "index.html"
    record_check(sec_smoke, dist_index.exists() and dist_index.stat().st_size > 0, "Vite production bundle dist/index.html exists")

    # Live Socket Smoke Test: Start Uvicorn on an ephemeral port and test real HTTP requests
    server_port = 8011
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=server_port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run)
    t.daemon = True
    t.start()

    time.sleep(1.5)
    try:
        base_live = f"http://127.0.0.1:{server_port}"
        with httpx.Client(base_url=base_live, timeout=10.0) as http_client:
            # /health
            r_live_h = http_client.get("/health")
            record_check(sec_smoke, r_live_h.status_code == 200, "Live Socket: /health HTTP 200")

            # /models
            r_live_m = http_client.get("/models")
            record_check(sec_smoke, r_live_m.status_code == 200, "Live Socket: /models HTTP 200")

            # /benchmark
            r_live_b = http_client.get("/benchmark")
            record_check(sec_smoke, r_live_b.status_code == 200, "Live Socket: /benchmark HTTP 200")

            # /selection
            r_live_s = http_client.get("/selection")
            record_check(sec_smoke, r_live_s.status_code == 200, "Live Socket: /selection HTTP 200")

            # /noise
            r_live_n = http_client.get("/noise")
            record_check(sec_smoke, r_live_n.status_code == 200, "Live Socket: /noise HTTP 200")

            # /predict
            r_live_p = http_client.post("/predict", json={"model": "rf", "features": CANONICAL_TEST_SAMPLE})
            record_check(sec_smoke, r_live_p.status_code == 200, "Live Socket: /predict HTTP 200")
    finally:
        server.should_exit = True
        t.join(timeout=3.0)

    record_check(sec_smoke, not t.is_alive(), "Live Uvicorn server shut down cleanly")

    # ==================================================================
    # 8. Cross-Stage Consistency
    # ==================================================================
    print("\n--- 8. Cross-Stage Consistency ---")
    sec_cross = "Cross-stage consistency"

    # Verify Stage 2-12 files all exist
    stage_artifacts = [
        ("Stage 2 Preprocessor", "src/preprocessing/preprocessor.py"),
        ("Stage 3 Classical Models", "src/classical/models.py"),
        ("Stage 4 VQC", "src/quantum/quantum_model.py"),
        ("Stage 5 Resource Optimizer", "src/quantum/resource_optimizer.py"),
        ("Stage 6 Noise Robustness", "src/quantum/noise_robustness.py"),
        ("Stage 7 Reliability", "src/evaluation/reliability.py"),
        ("Stage 8 SHAP Explainer", "src/explainability/shap_explainer.py"),
        ("Stage 9 Adaptive Selector", "src/quantum/adaptive_selector.py"),
        ("Stage 10 Final Benchmark", "src/evaluation/final_benchmark.py"),
        ("Stage 11 FastAPI Backend", "backend/main.py"),
        ("Stage 12 React Frontend", "frontend/src/App.jsx"),
    ]
    for stage_name, file_rel in stage_artifacts:
        record_check(sec_cross, (PROJECT_ROOT / file_rel).exists(), f"{stage_name} file '{file_rel}' preserved and active")

    # Verify benchmark matches canonical values from Stage 10
    bench_models = {m["Model"]: m for m in bench["unified_benchmark_table"]}
    rf_orig = bench_models.get("Random Forest")
    record_check(sec_cross, rf_orig and abs(rf_orig["ROC-AUC"] - 0.688172) < 1e-5, "Stage 10 RF ROC-AUC (0.688172) preserved in API")

    cfg_a = bench_models.get("Hybrid VQC (Config A)")
    record_check(sec_cross, cfg_a and abs(cfg_a["ROC-AUC"] - 0.604839) < 1e-5, "Stage 10 Config A ROC-AUC (0.604839) preserved in API")

    cfg_b = bench_models.get("Hybrid VQC (Config B)")
    record_check(sec_cross, cfg_b and abs(cfg_b["ROC-AUC"] - 0.631720) < 1e-5, "Stage 10 Config B ROC-AUC (0.631720) preserved in API")

    # Verify zero-specificity limitation is documented in benchmark
    lims = bench.get("limitations", [])
    lim_text = " ".join([l.get("finding", "") + " " + l.get("impact", "") for l in lims])
    record_check(sec_cross, "0.0000" in lim_text or "specificity = 0.0" in lim_text, "Quantum zero-specificity limitation documented")

    # ==================================================================
    # 9. Security, Leakage & Safety Checks
    # ==================================================================
    print("\n--- 9. Security, Leakage & Safety Checks ---")
    sec_sec = "Security/hardcoding checks"

    # Check for secrets or API keys
    secret_pats = [r"api_key\s*=", r"secret\s*=", r"password\s*="]
    found_secret = False
    for code_dir in [PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend" / "src"]:
        for p in code_dir.glob("**/*"):
            if p.is_file() and p.suffix in (".py", ".js", ".jsx"):
                text = p.read_text()
                for pat in secret_pats:
                    if re.search(pat, text, re.IGNORECASE):
                        found_secret = True
    record_check(sec_sec, not found_secret, "Zero hardcoded secrets, API keys, or passwords across backend and frontend")

    # Check no eval/exec in backend or frontend
    found_unsafe = False
    for py_file in (PROJECT_ROOT / "backend").glob("**/*.py"):
        if py_file.name == "verify_backend.py":
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    found_unsafe = True
    record_check(sec_sec, not found_unsafe, "Zero dynamic eval() or exec() calls in backend source")

    # Check no clinical diagnosis assertion in predictions
    risk_output = r_pred.json()["risk_signal"]
    record_check(sec_sec, "patient has parkinson" not in risk_output.lower(), "Safe non-diagnostic risk signal wording confirmed")

    # ==================================================================
    # Final Output Summary
    # ==================================================================
    print("\n" + "=" * 50)
    print("FULL SYSTEM INTEGRATION VERIFICATION")
    print("=" * 50)
    for section, passed in section_status.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"{section}: {status_str}")

    print("\n" + "=" * 50)
    print(f"TOTAL CHECKS: {checks_passed}/{total_checks} PASSED")
    print("=" * 50)

    all_passed = all(section_status.values())
    return all_passed


if __name__ == "__main__":
    success = run_full_system_verification()
    sys.exit(0 if success else 1)
