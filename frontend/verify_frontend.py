"""
Automated Verification Suite for Stage 12 — React Frontend Layer.

Checks:
  1. Frontend build assets exist and compile without error.
  2. All required component files exist.
  3. API base URL is strictly centralized in api.js.
  4. Zero hardcoded experimental metrics (ROC-AUC, accuracy, scores, deltas) in React components.
  5. Ground truth 'status' and 'name' are never present in feature inputs.
  6. Zero secrets, passwords, or API keys in frontend code.
  7. Zero eval/exec or dangerous innerHTML injections.
  8. Research & clinical disclaimers are visible in JSX.
  9. Quantum zero-specificity limitation is explicitly documented.
 10. End-to-end integration test: FastAPI backend responds to all API calls consumed by frontend.
"""

import sys
import re
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRONTEND_DIR = PROJECT_ROOT / "frontend"

REQUIRED_COMPONENTS = [
    "Navbar.jsx",
    "RiskCard.jsx",
    "ModelSelector.jsx",
    "PredictionForm.jsx",
    "ReliabilityCard.jsx",
    "ExplanationCard.jsx",
    "BenchmarkTable.jsx",
    "ResourceComparison.jsx",
    "NoiseCard.jsx",
    "SelectionCard.jsx",
    "Footer.jsx",
]

PRESET_FEATURES = {
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


def run_checks():
    print("=" * 70)
    print("STAGE 12: FRONTEND VERIFICATION SUITE")
    print("=" * 70)

    checks_passed = 0
    total_checks = 0

    def assert_check(cond, desc):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if cond:
            checks_passed += 1
            print(f"  [PASS] Check {total_checks:02d}: {desc}")
        else:
            print(f"  [FAIL] Check {total_checks:02d}: {desc}")
            raise AssertionError(f"Failed check: {desc}")

    # Check 1: Production build succeeds
    build_dir = FRONTEND_DIR / "dist"
    index_html = build_dir / "index.html"
    assert_check(index_html.exists() and index_html.stat().st_size > 0, "Frontend production bundle built successfully (dist/index.html)")

    # Check 2: All required components exist
    components_dir = FRONTEND_DIR / "src" / "components"
    for comp in REQUIRED_COMPONENTS:
        p = components_dir / comp
        assert_check(p.exists(), f"Component file '{comp}' exists")

    # Check 3: Centralized API module
    api_js = FRONTEND_DIR / "src" / "api.js"
    assert_check(api_js.exists(), "src/api.js exists")
    api_text = api_js.read_text()
    assert_check("export const API_BASE_URL" in api_text, "API base URL is declared and centralized in api.js")
    for fn in ["getHealth", "getModels", "getBenchmark", "getSelection", "getNoise", "predict"]:
        assert_check(f"export async function {fn}" in api_text, f"API helper '{fn}' exported in api.js")

    # Check 4: No hardcoded experimental results in JSX components
    prohibited_hardcodes = [
        ("0.688172", "RF ROC-AUC"),
        ("0.604839", "Config A ROC-AUC"),
        ("0.631720", "Config B ROC-AUC"),
        ("0.021279", "Config A noise delta"),
        ("0.040912", "Config B noise delta"),
    ]
    for comp in REQUIRED_COMPONENTS:
        text = (components_dir / comp).read_text()
        for num, desc in prohibited_hardcodes:
            assert_check(num not in text, f"Component '{comp}' does NOT hardcode {desc} ({num})")

    # Check 5: No ground truth 'status' or 'name' inputs in PredictionForm
    form_text = (components_dir / "PredictionForm.jsx").read_text()
    assert_check('"status"' not in form_text and "'status'" not in form_text, "PredictionForm does NOT contain 'status' feature")
    assert_check('"name"' not in form_text and "'name'" not in form_text, "PredictionForm does NOT contain 'name' feature")

    # Check 6: No secrets or API keys
    secret_pats = [r"api_key\s*=", r"secret\s*=", r"password\s*="]
    found_secret = False
    for p in (FRONTEND_DIR / "src").glob("**/*"):
        if p.is_file() and p.suffix in (".js", ".jsx", ".css", ".html"):
            t = p.read_text()
            for pat in secret_pats:
                if re.search(pat, t, re.IGNORECASE):
                    found_secret = True
    assert_check(not found_secret, "Zero hardcoded secrets, API keys, or passwords in frontend source")

    # Check 7: No dangerous HTML injection or eval
    unsafe_pats = ["dangerouslySetInnerHTML", "eval(", "Function("]
    found_unsafe = False
    for p in (FRONTEND_DIR / "src").glob("**/*"):
        if p.is_file() and p.suffix in (".js", ".jsx"):
            t = p.read_text()
            for u in unsafe_pats:
                if u in t:
                    found_unsafe = True
    assert_check(not found_unsafe, "Zero dangerous HTML injections or eval calls in frontend code")

    # Check 8: Research & clinical disclaimers present
    app_text = (FRONTEND_DIR / "src" / "App.jsx").read_text()
    footer_text = (components_dir / "Footer.jsx").read_text()
    assert_check("not a medical diagnosis" in app_text.lower() or "not an fda" in app_text.lower(), "Clinical research disclaimer visible in App.jsx")
    assert_check("medical diagnostic" in footer_text.lower() and "disclaimer" in footer_text.lower(), "Regulatory disclaimer visible in Footer.jsx")

    # Check 9: Quantum zero-specificity limitation documented
    bench_text = (components_dir / "BenchmarkTable.jsx").read_text()
    assert_check("0% specificity" in bench_text or "specificity = 0.0" in bench_text or "0.0000" in bench_text, "Quantum 0% specificity limitation explicitly documented in BenchmarkTable")

    # Check 10: Safe risk wording in RiskCard
    risk_text = (components_dir / "RiskCard.jsx").read_text()
    assert_check("Patient has" not in risk_text, "No unsupported diagnostic claims in RiskCard")

    # Check 11: End-to-end API integration smoke test
    from backend.main import app as fastapi_app
    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)

    r_health = client.get("/health")
    assert_check(r_health.status_code == 200 and r_health.json()["status"] == "ok", "E2E: /health responds with status ok")

    r_models = client.get("/models")
    assert_check(r_models.status_code == 200 and len(r_models.json()["models"]) == 5, "E2E: /models responds with 5 models")

    r_bench = client.get("/benchmark")
    assert_check(r_bench.status_code == 200 and len(r_bench.json()["unified_benchmark_table"]) == 8, "E2E: /benchmark responds with 8 models")

    r_sel = client.get("/selection")
    assert_check(r_sel.status_code == 200 and "Config A" in r_sel.json()["selected_configuration"], "E2E: /selection responds with Config A selected")

    r_noise = client.get("/noise")
    assert_check(r_noise.status_code == 200 and r_noise.json()["noise_probability"] == 0.05, "E2E: /noise responds with p=0.05")

    # Prediction test using the exact frontend preset payload
    r_pred = client.post("/predict", json={"model": "rf", "features": PRESET_FEATURES})
    assert_check(r_pred.status_code == 200 and "risk_signal" in r_pred.json(), "E2E: /predict executes successfully with preset features")

    # Prediction test with Quantum Config A
    r_pred_q = client.post("/predict", json={"model": "vqc_2q", "features": PRESET_FEATURES})
    assert_check(r_pred_q.status_code == 200 and r_pred_q.json()["reliability"]["available"] is True, "E2E: Quantum prediction with reliability succeeds")

    print("\n" + "=" * 70)
    print(f"ALL {checks_passed}/{total_checks} FRONTEND VERIFICATION CHECKS PASSED.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
