import React from 'react';

export default function NoiseCard({ noiseData, loading, error, onRetry }) {
  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '30px' }}>
        <span className="spinner" style={{ width: '28px', height: '28px', marginBottom: '8px' }} />
        <div style={{ color: 'var(--text-secondary)' }}>Loading quantum noise analysis data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="error-banner">
          <span>Failed to load noise analysis: {error}</span>
          <button type="button" className="btn btn-secondary" onClick={onRetry}>Retry</button>
        </div>
      </div>
    );
  }

  if (!noiseData) {
    return null;
  }

  const {
    noise_model,
    noise_probability,
    backend_device,
    config_a_prob_delta,
    config_b_prob_delta,
    interpretation,
    disclaimer,
  } = noiseData;

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>⚡</span>
          <span>Physical Quantum Noise Robustness (Stage 6)</span>
        </div>
        <span className="brand-badge">Simulated NISQ Environment</span>
      </div>

      <div className="card-description">
        Assessment of quantum state decoherence and expectation value contraction under gate-level depolarizing quantum noise channels (p = {noise_probability}) on the {backend_device} density-matrix simulator.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
        <div className="metric-tile" style={{ textAlign: 'left', padding: '16px' }}>
          <div className="metric-tile-label">Config A (2 Qubits) Probability Shift</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-emerald)', fontFamily: 'var(--font-mono)' }}>
            Δp = {config_a_prob_delta !== undefined ? config_a_prob_delta.toFixed(4) : 'N/A'}
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Lower circuit depth (9) limits decoherence accumulation across gates.
          </div>
        </div>

        <div className="metric-tile" style={{ textAlign: 'left', padding: '16px' }}>
          <div className="metric-tile-label">Config B (4 Qubits) Probability Shift</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-amber)', fontFamily: 'var(--font-mono)' }}>
            Δp = {config_b_prob_delta !== undefined ? config_b_prob_delta.toFixed(4) : 'N/A'}
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Higher circuit depth (13) and 2× gate count induces ~92% larger probability contraction.
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px', marginBottom: '14px' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-cyan)', marginBottom: '4px' }}>
          Exact Noise Model Architecture:
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          {noise_model}
        </div>
      </div>

      <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
        <strong>Scientific Interpretation:</strong> {interpretation}
      </div>

      <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '10px' }}>
        <strong>Simulation Disclaimer:</strong> {disclaimer}
      </div>
    </div>
  );
}
