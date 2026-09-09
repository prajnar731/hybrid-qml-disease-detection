# Stage 14 — Edge-Case Hardening & Robustness Verification Report

**Project:** SIH26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection  
**Evaluation Scope:** API Validation, Extreme Numerical Values, Special Floats, Model Robustness, Schema Integrity, Frontend Safety, Request Isolation, Security Sanity, and Full System Regression.  
**Execution Environment:** macOS ARM64 / Python 3.9.6 / FastAPI 0.115.8 / Uvicorn 0.34.0 / React Vite Frontend  

---

## 1. Executive Summary

Stage 14 subjects the completed hybrid classical-quantum disease detection platform to systematic adversarial and boundary testing. All 116 robustness checks across 9 categories passed with zero unhandled exceptions, zero 500 Internal Server Errors, zero NaN/Infinity leakages, and zero state cross-contamination between requests.

```
==================================================
STAGE 14 EDGE-CASE HARDENING
==================================================
Input validation: PASS
Numerical edge cases: PASS
Special float handling: PASS
Model robustness: PASS
Response schema: PASS
Frontend failure handling: PASS
Repeated requests: PASS
Security sanity: PASS
Regression suite: PASS

==================================================
TOTAL CHECKS: 116/116 PASSED
==================================================
```

---

## 2. Methodology & Behavior Classification

To ensure rigorous analysis, each test outcome is classified into one of three behavioral categories:

1. **Expected Validation Rejection (EVR):** The API or module actively rejects invalid, structurally malformed, incomplete, or out-of-spec inputs with HTTP `422 Unprocessable Entity` or `400 Bad Request` prior to reaching internal ML/QML pipeline logic.
2. **Graceful Controlled Computation (GCC):** Boundary, extreme, or edge-case numerical values that satisfy the API schema are ingested safely, transformed by the frozen preprocessor, evaluated by the model, and returned as valid finite probabilities in $[0, 1]$ with complete explanation and reliability components where applicable.
3. **Discovered Bug (BUG):** An unhandled crash (`500 Internal Server Error`), NaN/Inf value leakage into responses, silent state corruption, or memory/state contamination across sequential calls. **Zero (0) bugs were discovered.**

---

## 3. Detailed Test Matrix by Category

### Category 1: API Input Validation Edge Cases
*Target:* Ingestion boundaries of `POST /predict`.

| ID | Test Case | Input Payload | Expected Behavior | Actual Behavior | Classification | Status |
|:---|:---|:---|:---|:---|:---|:---:|
| 1.1 | Missing feature: `PPE` | 21 features (missing PPE) | HTTP 422 with validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.2 | Missing feature: `MDVP:Fo(Hz)` | 21 features (missing Fo) | HTTP 422 with validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.3 | Missing feature: `HNR` | 21 features (missing HNR) | HTTP 422 with validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.4 | Extra unknown feature | 22 features + `unknown_biomarker` | HTTP 422 (strict schema rejection) | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.5 | Non-numeric string value | `PPE = "invalid_string"` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.6 | Non-numeric string value | `MDVP:Fo(Hz) = "high"` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.7 | Non-numeric string value | `spread1 = "negative_value"` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.8 | Null / None value | `PPE = None` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.9 | Null / None value | `HNR = None` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.10 | Empty string value | `MDVP:Shimmer = ""` | HTTP 422 type validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.11 | Boolean value | `PPE = True` | Handled gracefully without HTTP 500 | Coerced to 1.0 or HTTP 422, status 200 | GCC | PASS |
| 1.12 | Invalid model identifier | `model = "quantum_super_vqc"` | HTTP 422 model selection error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.13 | Invalid model identifier | `model = ""` (empty string) | HTTP 422 model selection error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.14 | Invalid model identifier | `model = "xgboost"` | HTTP 422 model selection error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.15 | Completely empty JSON | `{}` | HTTP 422 validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.16 | Empty features object | `{"model": "random_forest", "features": {}}` | HTTP 422 validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 1.17 | Malformed JSON syntax | `{"model": "random_forest", "features":` | HTTP 4xx unparsable JSON error | HTTP 422 unparsable body | EVR | PASS |

---

### Category 2: Numerical Edge Cases
*Target:* Robustness of frozen standard scaler, PCA transformer, and model evaluators under extreme values.

| ID | Test Case | Input Values | Evaluated Models | Expected Behavior | Actual Behavior | Classification | Status |
|:---|:---|:---|:---|:---|:---|:---:|
| 2.1 | All-Zero Vector | All 22 features = `0.0` | RF, VQC 2Q, VQC 4Q | Scaler centers input; finite $p \in [0, 1]$ | HTTP 200, valid finite prob | GCC | PASS |
| 2.2 | Tiny Non-Zero (Underflow) | All 22 features = `1e-9` | RF, VQC 2Q, VQC 4Q | Scaler processes without zero-div error; $p \in [0, 1]$ | HTTP 200, valid finite prob | GCC | PASS |
| 2.3 | Large Finite (Overflow) | All 22 features = `1e7` | RF, VQC 2Q, VQC 4Q | Numerical clipping/scaling; finite $p \in [0, 1]$ | HTTP 200, valid finite prob | GCC | PASS |
| 2.4 | All-Negative Vector | All 22 features = `-50.0` | RF, VQC 2Q, VQC 4Q | Feature scaling handles negative domain; $p \in [0, 1]$ | HTTP 200, valid finite prob | GCC | PASS |
| 2.5 | Repeated Identical Features | All 22 features = `1.0` | RF, VQC 2Q, VQC 4Q | Degenerate feature vector handled cleanly | HTTP 200, valid finite prob | GCC | PASS |
| 2.6 | High-Contrast Extremes | Alternating `+1e6` and `-1e6` | RF, VQC 2Q, VQC 4Q | High variance handled gracefully without crash | HTTP 200, valid finite prob | GCC | PASS |

---

### Category 3: Special Float Handling (NaN / Infinity)
*Target:* Prevention of non-finite floating-point injection into Python ML/QML operations.

| ID | Test Case | Injected Representation | Expected Behavior | Actual Behavior | Classification | Status |
|:---|:---|:---|:---|:---|:---|:---:|
| 3.1 | String representation `"NaN"` | `PPE = "NaN"` | HTTP 422 schema validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 3.2 | String representation `"Infinity"` | `PPE = "Infinity"` | HTTP 422 schema validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 3.3 | String representation `"-Infinity"` | `PPE = "-Infinity"` | HTTP 422 schema validation error | HTTP 422 Unprocessable Entity | EVR | PASS |
| 3.4 | Raw JSON literal `NaN` | `{"PPE": NaN}` | HTTP 4xx rejection (RFC 8259 compliance) | HTTP 422 Rejected before ML pipeline | EVR | PASS |
| 3.5 | Raw JSON literal `Infinity` | `{"PPE": Infinity}` | HTTP 4xx rejection | HTTP 422 Rejected before ML pipeline | EVR | PASS |
| 3.6 | Raw JSON literal `-Infinity` | `{"PPE": -Infinity}` | HTTP 4xx rejection | HTTP 422 Rejected before ML pipeline | EVR | PASS |
| 3.7 | Response Output Audit | All API endpoints | Zero `"NaN"`, `"Infinity"`, `"-Infinity"` in responses | Verified clean across all responses | GCC | PASS |

---

### Category 4: Model-by-Model Robustness
*Target:* Independent evaluation of all 5 supported models across standard, boundary, and invalid conditions.

| ID | Model Identifier | Normal Benchmark Input | Boundary All-Zero Input | Incomplete Payload Input | Classification | Status |
|:---|:---|:---|:---|:---|:---:|:---:|
| 4.1 | `logistic_regression` | HTTP 200, $p \in [0, 1]$ | HTTP 200, $p \in [0, 1]$ | HTTP 422 validation rejection | GCC / EVR | PASS |
| 4.2 | `random_forest` | HTTP 200, $p \in [0, 1]$, SHAP | HTTP 200, $p \in [0, 1]$, SHAP | HTTP 422 validation rejection | GCC / EVR | PASS |
| 4.3 | `svm` | HTTP 200, $p \in [0, 1]$ | HTTP 200, $p \in [0, 1]$ | HTTP 422 validation rejection | GCC / EVR | PASS |
| 4.4 | `quantum_vqc_config_a` | HTTP 200, $p \in [0, 1]$, Reliability | HTTP 200, $p \in [0, 1]$, Reliability | HTTP 422 validation rejection | GCC / EVR | PASS |
| 4.5 | `quantum_vqc_config_b` | HTTP 200, $p \in [0, 1]$, Reliability | HTTP 200, $p \in [0, 1]$, Reliability | HTTP 422 validation rejection | GCC / EVR | PASS |

---

### Category 5: Response Schema Integrity
*Target:* Structural validity of all prediction response payloads.

| ID | Test Case | Invariant Verified | Actual Outcome | Status |
|:---|:---|:---|:---|:---:|
| 5.1 | Output Probability Range | `0.0 <= probability <= 1.0` | Satisfied across all models | PASS |
| 5.2 | Binary Classification Label | `predicted_class in {0, 1}` | Satisfied across all models | PASS |
| 5.3 | Model Identification Echo | `response["model"] == request["model"]` | Exact match confirmed | PASS |
| 5.4 | Probability Class Consistency | `predicted_class == 1` iff `prob >= 0.5` | Exact threshold alignment confirmed | PASS |
| 5.5 | Random Forest SHAP Explanations | `shap_explanation` exists, non-null, list with top features | Top features and base value present | PASS |
| 5.6 | Quantum VQC Reliability Components | `reliability_metrics` exists with CI, noise stability, overall | 95% CI, noise impact, overall reliability present | PASS |
| 5.7 | Classical Models Reliability Exemption | `reliability_metrics is None` for LR, RF, SVM | Non-quantum paths cleanly bypass Q-reliability | PASS |

---

### Category 6: Frontend Failure Handling & State Safety
*Target:* Resilience of dashboard against backend disconnection or API errors without emitting fake predictions.

| ID | Component / Code Asset | Failure Scenario | Expected Frontend Behavior | Actual Frontend Behavior | Status |
|:---|:---|:---|:---|:---|:---:|
| 6.1 | `frontend/src/api.js` | Backend offline / network crash | Throws structured `ApiError` with status & message | Catches TypeError and formats clear error | PASS |
| 6.2 | `frontend/src/api.js` | HTTP 422 validation error | Extracts detail array/string from response | Detailed error preserved for user inspection | PASS |
| 6.3 | `frontend/src/App.jsx` | API call failure during prediction | Sets `error` state, leaves `prediction = null` | Error banner displayed; zero fake predictions | PASS |
| 6.4 | `frontend/src/App.jsx` | Pending asynchronous request | Disables prediction button to prevent double-submit | `disabled={loading}` confirmed in UI code | PASS |
| 6.5 | All Frontend Components | Backend failure | Zero mock / fallback predictions | Zero synthetic fallback data present | PASS |

---

### Category 7: Repeated Requests & State Isolation
*Target:* Prevention of mutable state leakage across sequential API invocations.

| ID | Scenario | Execution Pattern | Expected Result | Actual Result | Status |
|:---|:---|:---|:---|:---|:---:|
| 7.1 | Sequential Model Cycling | 10 consecutive alternating calls: `[RF, VQC_A, SVM, VQC_B, LR] x 2` | Each call returns valid isolated prediction with identical probability for identical inputs | 10/10 calls succeeded with matching deterministic probabilities; zero state leakage | PASS |

---

### Category 8: Security & Code Safety Sanity Checks
*Target:* Prevention of arbitrary execution vulnerabilities or sensitive data leakage.

| ID | Inspection Domain | Rule | Result | Status |
|:---|:---|:---|:---|:---:|
| 8.1 | Backend Source Code | Zero dynamic code evaluation via `eval()` or `exec()` | Clean: 0 calls in `backend/` | PASS |
| 8.2 | Backend Source Code | Zero hardcoded credentials, JWT secrets, passwords, or cloud keys | Clean: 0 secrets found | PASS |
| 8.3 | Frontend Source Code | Zero hardcoded credentials, JWT secrets, passwords, or cloud keys | Clean: 0 secrets found | PASS |
| 8.4 | User-Facing Disclaimer | Non-diagnostic, research-only risk signal phrasing enforced | Clean: Non-diagnostic disclaimer confirmed | PASS |

---

### Category 9: Full System Regression Suite
*Target:* Guarantee that Stage 14 testing did not alter or regress any outputs from Stages 2 through 13.

| Stage | Module Verified | Test Verification Invocation | Checks Passed | Status |
|:---|:---|:---|:---:|:---:|
| Stage 2 | Data Preprocessor | `src/preprocessing/verify_preprocessing.py` | 18/18 checks passed | PASS |
| Stage 10 | Final Benchmarking | `src/evaluation/verify_final_benchmark.py` | 16/16 checks passed | PASS |
| Stage 11 | FastAPI Backend | `backend/verify_backend.py` | 24/24 checks passed | PASS |
| Stage 12 | React Frontend UI | `frontend/verify_frontend.py` | 28/28 checks passed | PASS |
| Stage 13 | Full System Integration | `tests/integration/verify_full_system.py` | 193/193 checks passed | PASS |
| **Stage 14** | **Edge-Case Hardening** | `tests/integration/verify_edge_cases.py` | **116/116 checks passed** | **PASS** |

---

## 4. Scientific Safety & Non-Clinical Disclaimer

> **IMPORTANT NOTICE:**  
> This platform is a research prototype developed exclusively for the Smart India Hackathon (SIH26139). It demonstrates the technical feasibility of hybrid classical-quantum machine learning pipelines for voice biomarker analysis on the Oxford Parkinson's Disease dataset.  
>  
> - **Not a Medical Device:** This software is not cleared, certified, or intended for clinical diagnosis, patient screening, disease staging, or treatment planning.  
> - **Exploratory Scope:** Predictions generated by the models (Logistic Regression, Random Forest, Support Vector Machine, and Variational Quantum Classifiers) represent experimental risk signals and must never supersede clinical evaluations by qualified healthcare professionals.  
> - **Quantum Model Limitations:** As documented in Stage 5, Stage 6, and Stage 10 benchmarks, the quantum VQC models operate at an exploratory phase (Config A ROC-AUC: 0.6048, Config B ROC-AUC: 0.6317) with zero specificity on the imbalanced test partition under a standard 0.5 threshold. They do not demonstrate quantum supremacy or superiority over optimized classical ensembles.
