import React from 'react';

export default function ResourceComparison({ selectionData }) {
  if (!selectionData || !selectionData.evaluated_configurations) {
    return null;
  }

  const { evaluated_configurations } = selectionData;

  return (
    <div className="card" style={{ marginTop: '20px' }}>
      <div className="card-header">
        <div className="card-title">
          <span>⚛️</span>
          <span>Quantum Resource Allocation: Config A vs Config B</span>
        </div>
        <span className="brand-badge">Measured Hardware Footprint</span>
      </div>

      <div className="card-description">
        Direct experimental comparison of quantum hardware allocation, coherence time requirements (circuit depth), and runtime.
      </div>

      <div className="comparison-grid">
        {evaluated_configurations.map((cfg, idx) => {
          const isWinner = cfg.is_selected;

          return (
            <div key={cfg.model_name || idx} className={`config-card ${isWinner ? 'selected' : ''}`}>
              {isWinner && <span className="winner-badge">★ SELECTED CONFIG</span>}

              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '4px' }}>
                {cfg.model_name}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-cyan)', marginBottom: '14px', fontFamily: 'var(--font-mono)' }}>
                {cfg.feature_representation}
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Qubits Allocation</span>
                <span className="spec-value">{cfg.qubits} Qubits</span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Circuit Depth</span>
                <span className="spec-value">Depth {cfg.circuit_depth}</span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Trainable Parameters</span>
                <span className="spec-value">{cfg.trainable_circuit_params} circuit ({cfg.total_params} total)</span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Training Wall-Clock Time</span>
                <span className="spec-value">{cfg.training_time_sec ? cfg.training_time_sec.toFixed(2) : 'N/A'} s</span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Ideal ROC-AUC</span>
                <span className="spec-value" style={{ color: 'var(--text-cyan)' }}>
                  {cfg.roc_auc ? cfg.roc_auc.toFixed(4) : 'N/A'}
                </span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Noise Probability Shift (Δp)</span>
                <span className="spec-value" style={{ color: isWinner ? 'var(--text-emerald)' : 'var(--text-amber)' }}>
                  {cfg.noise_prob_delta ? cfg.noise_prob_delta.toFixed(4) : 'N/A'}
                </span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Multi-Objective Score</span>
                <span className="spec-value" style={{ color: 'var(--text-indigo)', fontWeight: 700 }}>
                  {cfg.selection_score ? cfg.selection_score.toFixed(4) : 'N/A'}
                </span>
              </div>

              <div className="metric-spec-row">
                <span className="spec-name">Pareto Status</span>
                <span className="spec-value" style={{ color: 'var(--text-emerald)' }}>
                  {cfg.is_pareto_optimal ? 'Non-Dominated (Pareto Optimal)' : 'Dominated'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
