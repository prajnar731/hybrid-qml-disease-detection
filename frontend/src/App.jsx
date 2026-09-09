import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import ModelSelector from './components/ModelSelector';
import PredictionForm, { PRESET_SAMPLE_PARKINSONS } from './components/PredictionForm';
import RiskCard from './components/RiskCard';
import ReliabilityCard from './components/ReliabilityCard';
import ExplanationCard from './components/ExplanationCard';
import BenchmarkTable from './components/BenchmarkTable';
import SelectionCard from './components/SelectionCard';
import ResourceComparison from './components/ResourceComparison';
import NoiseCard from './components/NoiseCard';
import Footer from './components/Footer';

import {
  getHealth,
  getModels,
  getBenchmark,
  getSelection,
  getNoise,
  predict,
} from './api';

export default function App() {
  // Backend connection state
  const [backendOnline, setBackendOnline] = useState(false);
  const [activeSection, setActiveSection] = useState('overview');

  // Models list
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('rf');

  // Input features (initialized with Parkinson's benchmark sample)
  const [features, setFeatures] = useState(PRESET_SAMPLE_PARKINSONS);

  // Prediction state
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState(null);

  // Dashboard sections data
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(true);
  const [benchmarkError, setBenchmarkError] = useState(null);

  const [selectionData, setSelectionData] = useState(null);
  const [selectionLoading, setSelectionLoading] = useState(true);
  const [selectionError, setSelectionError] = useState(null);

  const [noiseData, setNoiseData] = useState(null);
  const [noiseLoading, setNoiseLoading] = useState(true);
  const [noiseError, setNoiseError] = useState(null);

  // Load all initial API data
  const loadInitialData = useCallback(async () => {
    // 1. Health Check
    try {
      const health = await getHealth();
      if (health && health.status === 'ok') {
        setBackendOnline(true);
      }
    } catch {
      setBackendOnline(false);
    }

    // 2. Models Catalog
    try {
      const modelsResp = await getModels();
      if (modelsResp && modelsResp.models) {
        setModels(modelsResp.models);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    }

    // 3. Benchmarks (Stage 10)
    try {
      setBenchmarkLoading(true);
      const bData = await getBenchmark();
      setBenchmarkData(bData);
      setBenchmarkError(null);
    } catch (err) {
      setBenchmarkError(err.message);
    } finally {
      setBenchmarkLoading(false);
    }

    // 4. Adaptive Selection (Stage 9)
    try {
      setSelectionLoading(true);
      const sData = await getSelection();
      setSelectionData(sData);
      setSelectionError(null);
    } catch (err) {
      setSelectionError(err.message);
    } finally {
      setSelectionLoading(false);
    }

    // 5. Noise Robustness (Stage 6)
    try {
      setNoiseLoading(true);
      const nData = await getNoise();
      setNoiseData(nData);
      setNoiseError(null);
    } catch (err) {
      setNoiseError(err.message);
    } finally {
      setNoiseLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Form input handlers
  const handleFeatureChange = (featureName, value) => {
    setFeatures((prev) => ({
      ...prev,
      [featureName]: value === '' ? '' : parseFloat(value) || 0,
    }));
  };

  const handleLoadPreset = (presetObj) => {
    setFeatures(presetObj);
    setPredictError(null);
  };

  const handleSubmitPrediction = async (e) => {
    e.preventDefault();
    setPredictLoading(true);
    setPredictError(null);

    try {
      // Validate all 22 features are numeric
      const numericPayload = {};
      for (const [k, v] of Object.entries(features)) {
        if (v === '' || isNaN(Number(v))) {
          throw new Error(`Invalid numeric input for feature '${k}'`);
        }
        numericPayload[k] = Number(v);
      }

      const response = await predict({
        model: selectedModelId,
        features: numericPayload,
      });

      setPredictionResult(response);
    } catch (err) {
      setPredictError(err.message || 'Prediction failed. Please check inputs.');
    } finally {
      setPredictLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Navbar
        backendOnline={backendOnline}
        onNavigate={setActiveSection}
        activeSection={activeSection}
      />

      <main className="main-content">
        {/* HERO / OVERVIEW SECTION */}
        <section id="overview" className="hero-section">
          <div className="hero-pill">
            ⚛️ Smart India Hackathon • SIH Problem ID: SIH26139
          </div>
          <h1 className="hero-title">
            Hybrid Quantum Machine Learning Platform for{' '}
            <span className="hero-highlight">Early Disease Detection</span>
          </h1>
          <p className="hero-subtitle">
            An algorithmic research demonstration integrating Classical Machine Learning, Variational
            Quantum Classifiers (VQC), Multi-Signal Reliability Assessment, and SHAP Interpretability
            for vocal acoustic biomarker screening.
          </p>

          <div className="research-disclaimer-banner">
            <span className="disclaimer-icon">⚠️</span>
            <div>
              <strong>Experimental Research Prototype:</strong> This platform is designed solely for
              scientific benchmarking on the UCI Oxford Parkinson's dataset. It is not an FDA/CE cleared
              diagnostic device. Do not use for patient management or clinical treatment decisions.
            </div>
          </div>

          {/* End-to-End Pipeline Visualization */}
          <div className="pipeline-container">
            <div className="pipeline-node active">1. Biomedical Voice Features (22)</div>
            <span className="pipeline-arrow">→</span>
            <div className="pipeline-node active">2. Scaler & PCA Preprocessing</div>
            <span className="pipeline-arrow">→</span>
            <div className="pipeline-node active">3. Classical & VQC Quantum Models</div>
            <span className="pipeline-arrow">→</span>
            <div className="pipeline-node active">4. Gate Noise Simulation (p=0.05)</div>
            <span className="pipeline-arrow">→</span>
            <div className="pipeline-node active">5. Multi-Signal Reliability & SHAP</div>
            <span className="pipeline-arrow">→</span>
            <div className="pipeline-node active">6. Calibrated Screening Output</div>
          </div>
        </section>

        {/* PREDICTION & INTERACTIVE SCREENING SECTION */}
        <section id="prediction">
          <div className="section-header">
            <div className="section-tag">Interactive Inference</div>
            <h2 className="section-title">Single-Sample Disease Risk Screening</h2>
          </div>

          <div className="prediction-grid">
            {/* Left: Input Form & Model Selector */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <span>⚙️</span>
                  <span>Input Parameters & Model Selection</span>
                </div>
              </div>

              <ModelSelector
                models={models}
                selectedModelId={selectedModelId}
                onSelectModel={setSelectedModelId}
              />

              <PredictionForm
                features={features}
                onChangeFeature={handleFeatureChange}
                onLoadPreset={handleLoadPreset}
                onSubmit={handleSubmitPrediction}
                loading={predictLoading}
                error={predictError}
              />
            </div>

            {/* Right: Results, Reliability & SHAP */}
            <div>
              <RiskCard predictionResult={predictionResult} />

              {predictionResult && (
                <>
                  <ReliabilityCard reliability={predictionResult.reliability} />
                  <ExplanationCard explanation={predictionResult.explanation} />
                </>
              )}
            </div>
          </div>
        </section>

        {/* BENCHMARKS SECTION */}
        <section id="benchmark">
          <div className="section-header">
            <div className="section-tag">Empirical Evaluation</div>
            <h2 className="section-title">Stage 10 Consolidated Benchmarks</h2>
          </div>

          <BenchmarkTable
            benchmarkData={benchmarkData}
            loading={benchmarkLoading}
            error={benchmarkError}
            onRetry={loadInitialData}
          />
        </section>

        {/* QUANTUM RESOURCE OPTIMIZATION SECTION */}
        <section id="optimization">
          <div className="section-header">
            <div className="section-tag">Algorithmic Innovation</div>
            <h2 className="section-title">Adaptive Quantum Resource Selection</h2>
          </div>

          <SelectionCard
            selectionData={selectionData}
            loading={selectionLoading}
            error={selectionError}
            onRetry={loadInitialData}
          />

          <ResourceComparison selectionData={selectionData} />
        </section>

        {/* NOISE ROBUSTNESS SECTION */}
        <section id="noise">
          <div className="section-header">
            <div className="section-tag">Physical Simulation</div>
            <h2 className="section-title">Quantum Noise Robustness Analysis</h2>
          </div>

          <NoiseCard
            noiseData={noiseData}
            loading={noiseLoading}
            error={noiseError}
            onRetry={loadInitialData}
          />
        </section>
      </main>

      <Footer />
    </div>
  );
}
