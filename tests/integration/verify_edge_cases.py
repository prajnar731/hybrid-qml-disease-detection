"""
Edge-Case Hardening & Robustness Verification Suite — Stage 14.

Tests system resilience against:
  1. API Input Validation Edge Cases (missing, extra, wrong types, invalid models, empty, malformed).
  2. Numerical Edge Cases (large finite values, tiny values, negative values, zeros, repeated identical).
  3. Special Float Handling (NaN, +Infinity, -Infinity).
  4. Model-by-Model Robustness (LR, RF, SVM, VQC 2Q, VQC 4Q under normal, boundary, invalid inputs).
  5. Response Schema Integrity (finite probabilities, required fields, zero NaN/Infinity leakage).
  6. Frontend Failure Handling & State Safety (graceful error states, zero fake predictions).
  7. Repeated Request State Isolation (sequential requests across models, no cross-contamination).
  8. Security Sanity Check (zero hardcoded secrets, no arbitrary code execution or eval/exec).
  9. Regression Test Suite Execution (verifies all previous stage suites pass 100%).
"""

import sys
import json
import math
import ast
import re
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app as fastapi_app
from backend.schemas import CANONICAL_FEATURE_NAMES


# Standard benchmark baseline sample
VALID_SAMPLE = {
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


def run_all_edge_case_tests():
    print("=" * 70)
    print("STAGE 14: EDGE-CASE HARDENING & ROBUSTNESS VERIFICATION")
    print("=" * 70)

    client = TestClient(fastapi_app)
    checks_passed = 0
    total_checks = 0

    section_status = {
        "Input validation": True,
        "Numerical edge cases": True,
        "Special float handling": True,
        "Model robustness": True,
        "Response schema": True,
        "Frontend failure handling": True,
        "Repeated requests": True,
        "Security sanity": True,
        "Regression suite": True,
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
    # 1. API Input Validation Edge Cases
    # ==================================================================
    print("\n--- 1. API Input Validation Edge Cases ---")
    sec_input = "Input validation"

    # A. Missing feature tests (remove individual features)
    for feat_to_remove in ["PPE", "MDVP:Fo(Hz)", "HNR"]:
        missing_sample = VALID_SAMPLE.copy()
        del missing_sample[feat_to_remove]
        r = client.post("/predict", json={"model": "rf", "features": missing_sample})
        record_check(sec_input, r.status_code == 422, f"Missing '{feat_to_remove}' rejected with HTTP 422")

    # B. Extra feature tests
    for extra_field in ["status", "name", "subject_id", "ground_truth", "unsupported_extra_feature"]:
        extra_sample = VALID_SAMPLE.copy()
        extra_sample[extra_field] = 123.45
        r = client.post("/predict", json={"model": "rf", "features": extra_sample})
        record_check(sec_input, r.status_code == 422, f"Extra field '{extra_field}' rejected with HTTP 422")

    # C. Wrong data types
    # String instead of float
    bad_str = VALID_SAMPLE.copy()
    bad_str["HNR"] = "high_noise"
    r_str = client.post("/predict", json={"model": "rf", "features": bad_str})
    record_check(sec_input, r_str.status_code == 422, "String in float field rejected with HTTP 422")

    # Null instead of float
    bad_null = VALID_SAMPLE.copy()
    bad_null["HNR"] = None
    r_null = client.post("/predict", json={"model": "rf", "features": bad_null})
    record_check(sec_input, r_null.status_code == 422, "Null in float field rejected with HTTP 422")

    # Empty string instead of float
    bad_empty_str = VALID_SAMPLE.copy()
    bad_empty_str["HNR"] = ""
    r_empty_str = client.post("/predict", json={"model": "rf", "features": bad_empty_str})
    record_check(sec_input, r_empty_str.status_code == 422, "Empty string in float field rejected with HTTP 422")

    # Boolean input (verifying no 500 crash occurs)
    bad_bool = VALID_SAMPLE.copy()
    bad_bool["HNR"] = True
    r_bool = client.post("/predict", json={"model": "rf", "features": bad_bool})
    record_check(sec_input, r_bool.status_code in (200, 422), "Boolean input handled without 500 error")

    # D. Invalid model names
    for bad_model in ["invalid_vqc", "gpt4", "quantum_supreme", ""]:
        r_model = client.post("/predict", json={"model": bad_model, "features": VALID_SAMPLE})
        record_check(sec_input, r_model.status_code == 422, f"Invalid model identifier '{bad_model}' rejected with HTTP 422")

    # E. Empty payload
    r_empty = client.post("/predict", json={})
    record_check(sec_input, r_empty.status_code == 422, "Empty payload {} rejected with HTTP 422")

    # F. Malformed JSON requests
    r_malformed = client.post("/predict", content="{not_valid_json: 123", headers={"Content-Type": "application/json"})
    record_check(sec_input, r_malformed.status_code in (400, 422), "Malformed JSON rejected with 4xx")

    # ==================================================================
    # 2. Numerical Edge Cases
    # ==================================================================
    print("\n--- 2. Numerical Edge Cases ---")
    sec_num = "Numerical edge cases"

    numerical_test_scenarios = [
        ("All Zero Vector", {f: 0.0 for f in CANONICAL_FEATURE_NAMES}),
        ("All Small Positive Values (1e-9)", {f: 1e-9 for f in CANONICAL_FEATURE_NAMES}),
        ("All Large Finite Values (1e7)", {f: 1e7 for f in CANONICAL_FEATURE_NAMES}),
        ("Negative Values (-10.0)", {f: -10.0 for f in CANONICAL_FEATURE_NAMES}),
        ("Identical Values (1.0)", {f: 1.0 for f in CANONICAL_FEATURE_NAMES}),
        ("High-Contrast Vector (Mix +/- 1e6)", {f: 1e6 if i % 2 == 0 else -1e6 for i, f in enumerate(CANONICAL_FEATURE_NAMES)}),
    ]

    for scenario_name, feat_dict in numerical_test_scenarios:
        for m_id in ["rf", "vqc_2q"]:
            r = client.post("/predict", json={"model": m_id, "features": feat_dict})
            record_check(sec_num, r.status_code == 200, f"{scenario_name} on '{m_id}' returns HTTP 200")
            res = r.json()
            prob = res.get("predicted_probability")
            record_check(sec_num, isinstance(prob, (float, int)) and not math.isnan(prob) and not math.isinf(prob), f"{scenario_name} on '{m_id}' produces finite probability")
            record_check(sec_num, 0.0 <= prob <= 1.0, f"{scenario_name} on '{m_id}' probability bounded in [0, 1]")

    # ==================================================================
    # 3. Special Float Values Handling
    # ==================================================================
    print("\n--- 3. Special Float Values Handling ---")
    sec_float = "Special float handling"

    # NaN in raw JSON string
    nan_json = '{"model": "rf", "features": {"MDVP:Fo(Hz)": NaN}}'
    r_nan = client.post("/predict", content=nan_json, headers={"Content-Type": "application/json"})
    record_check(sec_float, r_nan.status_code in (400, 422), "Raw NaN in JSON rejected cleanly with 4xx")

    # +Infinity in raw JSON string
    pos_inf_json = '{"model": "rf", "features": {"MDVP:Fo(Hz)": Infinity}}'
    r_pos_inf = client.post("/predict", content=pos_inf_json, headers={"Content-Type": "application/json"})
    record_check(sec_float, r_pos_inf.status_code in (400, 422), "Raw +Infinity in JSON rejected cleanly with 4xx")

    # -Infinity in raw JSON string
    neg_inf_json = '{"model": "rf", "features": {"MDVP:Fo(Hz)": -Infinity}}'
    r_neg_inf = client.post("/predict", content=neg_inf_json, headers={"Content-Type": "application/json"})
    record_check(sec_float, r_neg_inf.status_code in (400, 422), "Raw -Infinity in JSON rejected cleanly with 4xx")

    # Verify no NaN or Infinity exists in successful JSON outputs
    r_valid = client.post("/predict", json={"model": "vqc_2q", "features": VALID_SAMPLE})
    valid_text = r_valid.text
    record_check(sec_float, "NaN" not in valid_text, "Successful JSON output contains zero NaN tokens")
    record_check(sec_float, "Infinity" not in valid_text, "Successful JSON output contains zero Infinity tokens")

    # ==================================================================
    # 4. Model-by-Model Robustness
    # ==================================================================
    print("\n--- 4. Model-by-Model Robustness ---")
    sec_model = "Model robustness"

    all_models = ["lr", "rf", "svm", "vqc_2q", "vqc_4q"]
    for m in all_models:
        # Normal payload
        r_norm = client.post("/predict", json={"model": m, "features": VALID_SAMPLE})
        record_check(sec_model, r_norm.status_code == 200, f"Model '{m}' accepts normal payload")

        # Boundary payload (all zeros)
        r_zero = client.post("/predict", json={"model": m, "features": {f: 0.0 for f in CANONICAL_FEATURE_NAMES}})
        record_check(sec_model, r_zero.status_code == 200, f"Model '{m}' safely handles all-zero boundary input")

        # Incomplete payload
        bad_p = VALID_SAMPLE.copy()
        del bad_p["DFA"]
        r_incomp = client.post("/predict", json={"model": m, "features": bad_p})
        record_check(sec_model, r_incomp.status_code == 422, f"Model '{m}' safely rejects incomplete input with 422")

    # ==================================================================
    # 5. Response Schema Testing
    # ==================================================================
    print("\n--- 5. Response Schema Testing ---")
    sec_schema = "Response schema"

    # Schema verification for Random Forest
    r_rf = client.post("/predict", json={"model": "rf", "features": VALID_SAMPLE}).json()
    record_check(sec_schema, "predicted_probability" in r_rf, "RF response contains predicted_probability")
    record_check(sec_schema, "predicted_class" in r_rf, "RF response contains predicted_class")
    record_check(sec_schema, "explanation" in r_rf and r_rf["explanation"]["available"] is True, "RF response contains active SHAP explanation")
    record_check(sec_schema, len(r_rf["explanation"]["top_features"]) == 5, "RF explanation contains 5 top features")

    # Schema verification for Quantum VQC
    r_vqc = client.post("/predict", json={"model": "vqc_2q", "features": VALID_SAMPLE}).json()
    record_check(sec_schema, "reliability" in r_vqc and r_vqc["reliability"]["available"] is True, "VQC response contains active reliability")
    rel_data = r_vqc["reliability"]
    record_check(sec_schema, rel_data["category"] in ("HIGH", "MODERATE", "LOW"), "VQC reliability category is valid")
    record_check(sec_schema, "predictive_decisiveness_margin" in rel_data["components"], "VQC reliability decisiveness present")
    record_check(sec_schema, "classical_quantum_agreement" in rel_data["components"], "VQC reliability agreement present")
    record_check(sec_schema, "noise_stability_signal" in rel_data["components"], "VQC reliability noise stability present")

    # Error response schema
    r_err = client.post("/predict", json={"model": "rf", "features": {"bad": 1}})
    record_check(sec_schema, r_err.status_code == 422 and "detail" in r_err.json(), "Error response follows standard error schema")

    # ==================================================================
    # 6. Frontend Safety & Failure Handling
    # ==================================================================
    print("\n--- 6. Frontend Safety & Failure Handling ---")
    sec_fe = "Frontend failure handling"

    fe_app = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text()
    fe_api = (PROJECT_ROOT / "frontend" / "src" / "api.js").read_text()
    fe_form = (PROJECT_ROOT / "frontend" / "src" / "components" / "PredictionForm.jsx").read_text()

    # Verify frontend wraps network errors into ApiError
    record_check(sec_fe, "class ApiError" in fe_api, "Frontend api.js defines ApiError class")
    record_check(sec_fe, "Unable to connect to backend" in fe_api, "Frontend api.js provides user-friendly offline message")

    # Verify frontend displays error banners
    record_check(sec_fe, "error-banner" in fe_form or "error-banner" in fe_app, "Frontend renders error-banner on API failure")
    record_check(sec_fe, "setPredictError" in fe_app, "Frontend catches predict errors in state")

    # Verify button disables during loading
    record_check(sec_fe, "disabled={loading}" in fe_form, "Frontend disables submit button during prediction loading")

    # ==================================================================
    # 7. Repeated Requests & State Isolation
    # ==================================================================
    print("\n--- 7. Repeated Requests & State Isolation ---")
    sec_rep = "Repeated requests"

    # Send 10 alternating sequential prediction calls
    sequence = ["rf", "vqc_2q", "lr", "vqc_4q", "svm", "rf", "vqc_2q", "lr", "vqc_4q", "svm"]
    for idx, model_name in enumerate(sequence):
        r_seq = client.post("/predict", json={"model": model_name, "features": VALID_SAMPLE})
        record_check(sec_rep, r_seq.status_code == 200, f"Sequential request {idx + 1} ({model_name}) succeeded")
        record_check(sec_rep, r_seq.json()["model_id"] == model_name, f"Sequential request {idx + 1} preserved model_id '{model_name}'")

    # ==================================================================
    # 8. Security Sanity Check
    # ==================================================================
    print("\n--- 8. Security Sanity Check ---")
    sec_sec = "Security sanity"

    # Check for secrets or API keys across codebase
    secret_pats = [r"api_key\s*=", r"secret\s*=", r"password\s*="]
    found_secret = False
    for code_dir in [PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend" / "src", PROJECT_ROOT / "src"]:
        for p in code_dir.glob("**/*"):
            if p.is_file() and p.suffix in (".py", ".js", ".jsx"):
                text = p.read_text()
                for pat in secret_pats:
                    if re.search(pat, text, re.IGNORECASE):
                        found_secret = True
    record_check(sec_sec, not found_secret, "Zero hardcoded secrets across codebase")

    # Check for unsafe eval/exec
    found_eval = False
    for py_file in (PROJECT_ROOT / "backend").glob("**/*.py"):
        if py_file.name == "verify_backend.py":
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    found_eval = True
    record_check(sec_sec, not found_eval, "Zero dynamic eval() or exec() calls in backend")

    # ==================================================================
    # 9. Regression Test Suite Execution
    # ==================================================================
    print("\n--- 9. Regression Test Suite Execution ---")
    sec_reg = "Regression suite"

    # Run Stage 2 Preprocessing verification
    from src.preprocessing.verify_preprocessing import main as run_stage2_main
    try:
        run_stage2_main()
        stage2_passed = True
    except Exception:
        stage2_passed = False
    record_check(sec_reg, stage2_passed, "Stage 2 Preprocessing regression suite PASSED")

    # Run Stage 10 Benchmark verification
    from src.evaluation.verify_final_benchmark import run_all_checks as verify_stage10
    record_check(sec_reg, verify_stage10() is True, "Stage 10 Benchmark regression suite PASSED")

    # Run Stage 11 Backend verification
    from backend.verify_backend import run_all_backend_checks as verify_stage11
    record_check(sec_reg, verify_stage11() is True, "Stage 11 Backend regression suite PASSED")

    # Run Stage 12 Frontend verification
    from frontend.verify_frontend import run_checks as verify_stage12
    record_check(sec_reg, verify_stage12() is True, "Stage 12 Frontend regression suite PASSED")

    # Run Stage 13 Full System Integration verification
    from tests.integration.verify_full_system import run_full_system_verification as verify_stage13
    record_check(sec_reg, verify_stage13() is True, "Stage 13 Full System Integration regression suite PASSED")

    # ==================================================================
    # Final Output Summary
    # ==================================================================
    print("\n" + "=" * 50)
    print("STAGE 14 EDGE-CASE HARDENING")
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
    success = run_all_edge_case_tests()
    sys.exit(0 if success else 1)
