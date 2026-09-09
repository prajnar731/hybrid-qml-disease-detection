import React from 'react';

// Example benchmark sample from test set for quick testing
export const PRESET_SAMPLE_PARKINSONS = {
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
};

export const PRESET_SAMPLE_HEALTHY = {
  "MDVP:Fo(Hz)": 197.076,
  "MDVP:Fhi(Hz)": 206.896,
  "MDVP:Flo(Hz)": 192.055,
  "MDVP:Jitter(%)": 0.00289,
  "MDVP:Jitter(Abs)": 0.00001,
  "MDVP:RAP": 0.00166,
  "MDVP:PPQ": 0.00168,
  "Jitter:DDP": 0.00498,
  "MDVP:Shimmer": 0.01098,
  "MDVP:Shimmer(dB)": 0.097,
  "Shimmer:APQ3": 0.00563,
  "Shimmer:APQ5": 0.00680,
  "MDVP:APQ": 0.00802,
  "Shimmer:DDA": 0.01689,
  "NHR": 0.00339,
  "HNR": 26.775,
  "RPDE": 0.422229,
  "DFA": 0.741367,
  "spread1": -7.348300,
  "spread2": 0.177551,
  "D2": 1.743867,
  "PPE": 0.085569,
};

const FEATURE_GROUPS = [
  {
    title: '1. Fundamental Frequency Measures',
    features: [
      { name: 'MDVP:Fo(Hz)', label: 'Average Fo (Hz)' },
      { name: 'MDVP:Fhi(Hz)', label: 'Max Fhi (Hz)' },
      { name: 'MDVP:Flo(Hz)', label: 'Min Flo (Hz)' },
    ],
  },
  {
    title: '2. Jitter (Frequency Variation) Measures',
    features: [
      { name: 'MDVP:Jitter(%)', label: 'Jitter (%)' },
      { name: 'MDVP:Jitter(Abs)', label: 'Jitter (Abs)' },
      { name: 'MDVP:RAP', label: 'RAP' },
      { name: 'MDVP:PPQ', label: 'PPQ' },
      { name: 'Jitter:DDP', label: 'Jitter DDP' },
    ],
  },
  {
    title: '3. Shimmer (Amplitude Variation) Measures',
    features: [
      { name: 'MDVP:Shimmer', label: 'Local Shimmer' },
      { name: 'MDVP:Shimmer(dB)', label: 'Shimmer (dB)' },
      { name: 'Shimmer:APQ3', label: 'APQ3' },
      { name: 'Shimmer:APQ5', label: 'APQ5' },
      { name: 'MDVP:APQ', label: 'MDVP APQ' },
      { name: 'Shimmer:DDA', label: 'Shimmer DDA' },
    ],
  },
  {
    title: '4. Noise & Nonlinear Dynamical Measures',
    features: [
      { name: 'NHR', label: 'NHR' },
      { name: 'HNR', label: 'HNR' },
      { name: 'RPDE', label: 'RPDE' },
      { name: 'DFA', label: 'DFA' },
      { name: 'spread1', label: 'Spread 1' },
      { name: 'spread2', label: 'Spread 2' },
      { name: 'D2', label: 'D2 (Correlation Dim)' },
      { name: 'PPE', label: 'PPE' },
    ],
  },
];

export default function PredictionForm({
  features,
  onChangeFeature,
  onLoadPreset,
  onSubmit,
  loading,
  error,
}) {
  const handleInputChange = (featureName, value) => {
    onChangeFeature(featureName, value);
  };

  return (
    <form onSubmit={onSubmit} className="prediction-form">
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => onLoadPreset(PRESET_SAMPLE_PARKINSONS)}
        >
          📋 Load Elevated Risk Sample
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => onLoadPreset(PRESET_SAMPLE_HEALTHY)}
        >
          📋 Load Low Risk Sample
        </button>
      </div>

      {FEATURE_GROUPS.map((group) => (
        <div key={group.title} className="feature-group-box">
          <div className="group-title">{group.title}</div>
          <div className="feature-inputs-grid">
            {group.features.map((f) => (
              <div key={f.name} className="input-wrapper">
                <label className="input-label" htmlFor={f.name} title={f.name}>
                  {f.name}
                </label>
                <input
                  id={f.name}
                  name={f.name}
                  type="number"
                  step="any"
                  className="feature-input"
                  value={features[f.name] !== undefined ? features[f.name] : ''}
                  onChange={(e) => handleInputChange(f.name, e.target.value)}
                  required
                />
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <>
              <span className="spinner" />
              <span>Evaluating Quantum/Classical Circuit...</span>
            </>
          ) : (
            <>
              <span>⚡</span>
              <span>Execute Disease Screening Inference</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
}
