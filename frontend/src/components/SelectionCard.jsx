import React from 'react';

export default function SelectionCard({ selectionData, loading, error, onRetry }) {
  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '30px' }}>
        <span className="spinner" style={{ width: '28px', height: '28px', marginBottom: '8px' }} />
        <div style={{ color: 'var(--text-secondary)' }}>Loading quantum resource selection data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="error-banner">
          <span>Failed to load adaptive selection: {error}</span>
          <button type="button" className="btn btn-secondary" onClick={onRetry}>Retry</button>
        </div>
      </div>
    );
  }

  if (!selectionData) {
    return null;
  }

  const {
    selected_configuration,
    selection_score,
    selection_weights,
    pareto_frontier,
    selection_rationale,
    evaluated_configurations,
  } = selectionData;

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>⚖️</span>
          <span>Adaptive Quantum Resource Selection (Stage 9)</span>
        </div>
        <span className="brand-badge">Multi-Objective Optimization</span>
      </div>

      <div className="card-description">
        Quantitative trade-off optimization balancing predictive performance, quantum hardware cost, and gate-level noise resilience.
      </div>

      <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: 'var(--radius-md)', padding: '16px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-cyan)', textTransform: 'uppercase', fontWeight: 700 }}>
              ★ Selected Quantum Operating Point
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#ffffff', marginTop: '2px' }}>
              {selected_configuration}
            </div>
          </div>
          <div>
            <span className="brand-badge" style={{ fontSize: '0.9rem', padding: '6px 12px' }}>
              Composite Score: {selection_score.toFixed(4)}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '10px' }}>
          {selection_rationale}
        </div>
      </div>

      {selection_weights && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '20px' }}>
          <div className="metric-tile" style={{ padding: '10px' }}>
            <div className="metric-tile-label">Performance Weight</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {(selection_weights.weight_performance * 100).toFixed(1)}%
            </div>
          </div>
          <div className="metric-tile" style={{ padding: '10px' }}>
            <div className="metric-tile-label">Resource Weight</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {(selection_weights.weight_resource * 100).toFixed(1)}%
            </div>
          </div>
          <div className="metric-tile" style={{ padding: '10px' }}>
            <div className="metric-tile-label">Noise Robustness Weight</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {(selection_weights.weight_noise * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {pareto_frontier && (
        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
          <strong>Empirical Pareto Frontier:</strong>
          <ul style={{ paddingLeft: '18px', marginTop: '4px' }}>
            {pareto_frontier.map((p, idx) => (
              <li key={idx} style={{ marginBottom: '2px' }}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
