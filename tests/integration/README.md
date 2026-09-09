# Stage 13: Full System Integration & End-to-End Testing

**Project:** Hybrid Quantum Machine Learning Platform for Early Disease Detection  
**SIH Problem ID:** SIH26139  
**Stage:** 13 — System Integration & End-to-End Testing  

---

## 1. Overview & Verification Scope

Stage 13 verifies the complete, uninterrupted data and execution pipeline connecting all frozen stages of the project:

```
React 18 Frontend (Vite)
       ↓  HTTP / REST JSON
FastAPI Backend (Uvicorn ASGI)
       ↓
Stage 2: ParkinsonsPreprocessor (StandardScaler + Optional PCA, No Leakage)
       ↓
Stage 3 & 4/5: Machine Learning & Variational Quantum Classifiers (VQC 2Q & 4Q)
       ↓
Stage 6: Gate-Level Depolarizing Quantum Noise Simulation (p=0.05 on default.mixed)
       ↓
Stage 7 & 8: Multi-Signal Reliability Assessment & Local SHAP TreeExplainer Attributions
       ↓
Stage 9 & 10: Adaptive Resource Optimization & Unified Empirical Benchmarks
       ↓
Calibrated JSON Response Serialization & Frontend Rendering
```

---

## 2. Tested Endpoints & Integration Surface

All endpoints are validated through both direct ASGI test client execution and real background socket tests on localhost:

1. **`GET /health`**: Verifies service status and identifier.
2. **`GET /models`**: Verifies catalog of 5 models (`rf`, `lr`, `svm`, `vqc_2q`, `vqc_4q`) with qubit allocations and circuit depths.
3. **`GET /benchmark`**: Consolidates 8 models from Stage 10, metric leaders, and explicit research boundaries.
4. **`GET /selection`**: Validates Stage 9 equal-weight multi-objective selection favoring Config A (score: 0.6667).
5. **`GET /noise`**: Validates Stage 6 gate depolarizing noise ($p=0.05$) probability deltas ($\Delta p = 0.0213$ vs $0.0409$).
6. **`GET /docs` & `/openapi.json`**: Verifies OpenAPI schema generation.
7. **`POST /predict`**: Verifies single-sample inference across all 5 models with:
   - Valid probability output in $[0, 1]$
   - Safe non-diagnostic risk signal wording (*"Elevated/Low Parkinson's risk signal detected."*)
   - Local SHAP explanation for Random Forest
   - Multi-signal reliability assessment for quantum models (Decisiveness, Agreement, Noise Stability)

---

## 3. How to Run the Verification Suite

Run the full system integration test suite from the project root:

```bash
python3 tests/integration/verify_full_system.py
```

To run all historical stage verification suites:

```bash
python3 src/preprocessing/verify_preprocessing.py
python3 src/classical/verify_classical.py
python3 src/quantum/verify_quantum.py
python3 src/quantum/verify_noise_robustness.py
python3 src/evaluation/verify_reliability.py
python3 src/explainability/verify_shap.py
python3 src/quantum/verify_adaptive_selector.py
python3 src/evaluation/verify_final_benchmark.py
python3 backend/verify_backend.py
python3 frontend/verify_frontend.py
```

---

## 4. Documented Limitations & Research Boundaries

1. **Software Integration vs Clinical Validation:**
   - This test suite confirms that the software pipeline, APIs, and user interfaces function end-to-end without runtime errors, type mismatches, or data leakage.
   - **This is NOT a clinical trial, clinical study, or medical validation.**
2. **Quantum Threshold Limitation:**
   - Both VQC configurations (Config A and Config B) classify all 43 test samples as positive at the default 0.5 threshold, resulting in $100\%$ recall and $0\%$ specificity on this test cohort.
   - This limitation is documented and displayed transparently in the API and UI; high sensitivity does NOT imply clinical efficacy.
3. **Classical vs. Quantum Performance:**
   - Classical Random Forest on 22 original features achieves higher discriminative ranking ($\text{ROC-AUC} = 0.6882$) than the quantum models ($0.6048$ and $0.6317$). No quantum advantage or supremacy is claimed.
4. **Non-Diagnostic Scope:**
   - The platform is strictly an engineering research prototype for vocal biomarker analysis. It does not provide medical diagnoses or treatment recommendations.
