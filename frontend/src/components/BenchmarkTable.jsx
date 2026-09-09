import React from 'react';

export default function BenchmarkTable({ benchmarkData, loading, error, onRetry }) {
  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
        <span className="spinner" style={{ width: '32px', height: '32px', marginBottom: '12px' }} />
        <div style={{ color: 'var(--text-secondary)' }}>Loading unified benchmark data from API...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="error-banner">
          <span>Failed to load benchmarks: {error}</span>
          <button type="button" className="btn btn-secondary" onClick={onRetry}>Retry</button>
        </div>
      </div>
    );
  }

  if (!benchmarkData || !benchmarkData.unified_benchmark_table) {
    return null;
  }

  const { unified_benchmark_table, metric_leaders } = benchmarkData;

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>📈</span>
          <span>Unified Experimental Benchmarks (Stage 10)</span>
        </div>
        <span className="brand-badge">Subject-Aware Test Cohort (N=43)</span>
      </div>

      <div className="card-description">
        Consolidated performance across 8 classical and quantum models evaluated on 7 strictly held-out subjects (zero identity leakage).
      </div>

      {metric_leaders && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          {metric_leaders.best_roc_auc && (
            <div className="metric-tile" style={{ textAlign: 'left', padding: '12px' }}>
              <div className="metric-tile-label">Leading ROC-AUC</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-cyan)', fontFamily: 'var(--font-mono)' }}>
                {metric_leaders.best_roc_auc.value.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#ffffff', fontWeight: 600 }}>
                {metric_leaders.best_roc_auc.model}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {metric_leaders.best_roc_auc.feature_representation}
              </div>
            </div>
          )}

          {metric_leaders.best_specificity && (
            <div className="metric-tile" style={{ textAlign: 'left', padding: '12px' }}>
              <div className="metric-tile-label">Leading Specificity</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-emerald)', fontFamily: 'var(--font-mono)' }}>
                {metric_leaders.best_specificity.value.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#ffffff', fontWeight: 600 }}>
                {metric_leaders.best_specificity.model}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                2/12 true healthy subjects identified
              </div>
            </div>
          )}

          {metric_leaders.best_quantum_resource_efficiency && (
            <div className="metric-tile" style={{ textAlign: 'left', padding: '12px' }}>
              <div className="metric-tile-label">Quantum Efficiency Leader</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-indigo)', fontFamily: 'var(--font-mono)' }}>
                {metric_leaders.best_quantum_resource_efficiency.qubits} Qubits
              </div>
              <div style={{ fontSize: '0.75rem', color: '#ffffff', fontWeight: 600 }}>
                {metric_leaders.best_quantum_resource_efficiency.model}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                Depth {metric_leaders.best_quantum_resource_efficiency.circuit_depth} • 30.27s training
              </div>
            </div>
          )}
        </div>
      )}

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Representation</th>
              <th>Qubits</th>
              <th className="num-cell">Accuracy</th>
              <th className="num-cell">Precision</th>
              <th className="num-cell">Recall</th>
              <th className="num-cell">F1-Score</th>
              <th className="num-cell">ROC-AUC</th>
              <th className="num-cell">Specificity</th>
            </tr>
          </thead>
          <tbody>
            {unified_benchmark_table.map((row, idx) => {
              const isQuantum = row.Qubits !== '—';
              return (
                <tr key={`${row.Model}-${idx}`} className={isQuantum ? 'highlight-row' : ''}>
                  <td style={{ fontWeight: 600 }}>
                    {isQuantum ? '⚛️ ' : '💻 '}
                    {row.Model}
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{row['Feature Representation']}</td>
                  <td>{row.Qubits}</td>
                  <td className="num-cell">{row.Accuracy.toFixed(4)}</td>
                  <td className="num-cell">{row.Precision.toFixed(4)}</td>
                  <td className="num-cell">{row.Recall.toFixed(4)}</td>
                  <td className="num-cell">{row['F1-Score'].toFixed(4)}</td>
                  <td className="num-cell" style={{ fontWeight: 700, color: 'var(--text-cyan)' }}>
                    {row['ROC-AUC'].toFixed(4)}
                  </td>
                  <td className="num-cell" style={{ color: row.Specificity > 0 ? 'var(--text-emerald)' : 'var(--text-rose)' }}>
                    {row.Specificity.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="limitation-alert">
        <span style={{ fontSize: '1.4rem' }}>⚠️</span>
        <div>
          <strong>Critical Scientific Limitation (Quantum Threshold Behavior):</strong>
          <div style={{ marginTop: '4px' }}>
            Both current quantum configurations (Config A and Config B) classified every test sample as positive at the established 0.5 threshold, resulting in 100% sensitivity (Recall = 1.0000) but 0% specificity (0.0000) on this test cohort. High recall alone is <strong>NOT</strong> evidence of clinical effectiveness. Classical Random Forest trained on original features achieved higher discriminative ranking (ROC-AUC 0.6882) than both quantum models. No quantum advantage is claimed.
          </div>
        </div>
      </div>
    </div>
  );
}
