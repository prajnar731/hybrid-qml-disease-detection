import React from 'react';

export default function RiskCard({ predictionResult }) {
  if (!predictionResult) {
    return (
      <div className="card risk-card">
        <div className="card-header">
          <div className="card-title">
            <span>📊</span>
            <span>Screening Prediction Output</span>
          </div>
        </div>
        <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px 10px', fontSize: '0.9rem' }}>
          Select a model, enter 22 voice acoustic features, and click "Execute Disease Screening Inference" to generate real-time results.
        </div>
      </div>
    );
  }

  const {
    model_name,
    feature_representation,
    predicted_probability,
    predicted_class,
    risk_signal,
    decision_threshold,
    disclaimer,
  } = predictionResult;

  const isElevated = predicted_class === 1;
  const percentage = Math.round(predicted_probability * 100);

  return (
    <div className="card risk-card">
      <div className="card-header">
        <div className="card-title">
          <span>📊</span>
          <span>Screening Prediction Output</span>
        </div>
        <span className="brand-badge">{model_name}</span>
      </div>

      <div className={`risk-signal-banner ${isElevated ? 'elevated' : 'low'}`}>
        <div className="risk-icon-circle">{isElevated ? '⚠️' : '🛡️'}</div>
        <div>
          <div className="risk-title">{risk_signal}</div>
          <div className="risk-subtitle">
            Feature space: {feature_representation} • Threshold: {decision_threshold}
          </div>
        </div>
      </div>

      <div className="metrics-gauge-row">
        <div className="metric-tile">
          <div className="metric-tile-label">Model Probability</div>
          <div className="metric-tile-value">{predicted_probability.toFixed(4)}</div>
          <div className="progress-bar-container">
            <div
              className={`progress-bar-fill ${isElevated ? 'elevated' : 'low'}`}
              style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
            />
          </div>
        </div>

        <div className="metric-tile">
          <div className="metric-tile-label">Decision State</div>
          <div className="metric-tile-value" style={{ fontSize: '1.25rem', paddingTop: '6px' }}>
            {isElevated ? 'ELEVATED RISK' : 'LOW RISK'}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Class: {predicted_class} (Binary output)
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '10px' }}>
        <strong>Scientific Disclaimer:</strong> {disclaimer}
      </div>
    </div>
  );
}
