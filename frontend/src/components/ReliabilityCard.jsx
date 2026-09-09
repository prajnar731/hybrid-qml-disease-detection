import React from 'react';

export default function ReliabilityCard({ reliability }) {
  if (!reliability) {
    return null;
  }

  const { available, category, composite_score, components, disclaimer, cohort_calibration_note } = reliability;

  if (!available) {
    return (
      <div className="card" style={{ marginTop: '20px' }}>
        <div className="card-header">
          <div className="card-title">
            <span>🛡️</span>
            <span>Reliability-Aware Assessment</span>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Classical Baseline</span>
        </div>
        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
          Multi-signal quantum reliability assessment (Predictive Decisiveness, Cross-Model Agreement, Noise Stability) is evaluated specifically for hybrid quantum classifiers (Config A & Config B).
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginTop: '20px' }}>
      <div className="card-header">
        <div className="card-title">
          <span>🛡️</span>
          <span>Reliability-Aware Assessment (Stage 7)</span>
        </div>
        <span className={`reliability-badge ${category}`}>
          ● {category} RELIABILITY
        </span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Composite Research Score (R)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-cyan)' }}>
            {composite_score !== undefined ? composite_score.toFixed(4) : 'N/A'}
          </div>
        </div>
        <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-muted)', maxWidth: '300px' }}>
          Equal 1/3 weighting across Decisiveness, Agreement, and Gate Noise Stability.
        </div>
      </div>

      {components && (
        <div className="signals-breakdown-list">
          <div className="signal-item">
            <div className="signal-header">
              <span className="signal-name">1. Predictive Decisiveness (Decision Margin)</span>
              <span className="signal-value">{components.predictive_decisiveness_margin.toFixed(4)}</span>
            </div>
            <div className="signal-desc">
              Measures normalized distance from uninformative 0.5 classification boundary (2 × |p − 0.5|). Indicates decision margin, NOT correctness.
            </div>
          </div>

          <div className="signal-item">
            <div className="signal-header">
              <span className="signal-name">2. Classical-Quantum Agreement</span>
              <span className="signal-value">{components.classical_quantum_agreement.toFixed(4)}</span>
            </div>
            <div className="signal-desc">
              Concordance with paired classical Random Forest baseline on this sample (1 − |p_q − p_c|). Measures cross-architecture consistency.
            </div>
          </div>

          <div className="signal-item">
            <div className="signal-header">
              <span className="signal-name">3. Physical Noise Stability (p=0.05)</span>
              <span className="signal-value">{components.noise_stability_signal.toFixed(4)}</span>
            </div>
            <div className="signal-desc">
              Resistance to gate-level depolarizing quantum noise on density matrix simulation (1 − |p_ideal − p_noisy|).
            </div>
          </div>
        </div>
      )}

      {cohort_calibration_note && (
        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
          ℹ️ {cohort_calibration_note}
        </div>
      )}

      <div style={{ fontSize: '0.76rem', color: 'var(--text-amber)', background: 'rgba(245, 158, 11, 0.08)', padding: '8px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
        <strong>Important Scientific Disclaimer:</strong> {disclaimer}
      </div>
    </div>
  );
}
