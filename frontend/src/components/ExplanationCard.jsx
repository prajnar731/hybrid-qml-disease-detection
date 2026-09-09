import React from 'react';

export default function ExplanationCard({ explanation }) {
  if (!explanation) {
    return null;
  }

  const { available, method, base_value, top_features, disclaimer } = explanation;

  if (!available) {
    return (
      <div className="card" style={{ marginTop: '20px' }}>
        <div className="card-header">
          <div className="card-title">
            <span>🔍</span>
            <span>SHAP Interpretability</span>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latent Space Model</span>
        </div>
        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
          Feature-level SHAP TreeExplainer attributions are configured for the 22 original biomedical features via Random Forest. Hybrid quantum classifiers operate over PCA-reduced quantum state rotations.
        </div>
      </div>
    );
  }

  // Find max absolute SHAP value for proportional bar scaling
  const maxAbsShap = top_features && top_features.length > 0
    ? Math.max(...top_features.map((f) => Math.abs(f.shap_value)), 0.001)
    : 1;

  return (
    <div className="card" style={{ marginTop: '20px' }}>
      <div className="card-header">
        <div className="card-title">
          <span>🔍</span>
          <span>SHAP Interpretability (Stage 8)</span>
        </div>
        <span className="brand-badge">{method}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
          Expected Base Value: <strong style={{ fontFamily: 'var(--font-mono)', color: '#ffffff' }}>{base_value !== undefined ? base_value.toFixed(4) : 'N/A'}</strong>
        </div>
        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
          Top Influential Biomedical Acoustic Features
        </div>
      </div>

      {top_features && top_features.map((feat) => {
        const isRiskIncrease = feat.shap_value > 0;
        const barWidth = Math.min(100, Math.round((Math.abs(feat.shap_value) / maxAbsShap) * 100));

        return (
          <div key={feat.feature} className="shap-bar-container">
            <div className="shap-bar-header">
              <span className="shap-feat-name">
                {isRiskIncrease ? '▲ ' : '▼ '}
                {feat.feature}
              </span>
              <span className="shap-feat-val">
                SHAP: <strong style={{ color: isRiskIncrease ? 'var(--text-rose)' : 'var(--text-emerald)' }}>
                  {feat.shap_value > 0 ? `+${feat.shap_value.toFixed(4)}` : feat.shap_value.toFixed(4)}
                </strong>{' '}
                (Raw: {feat.feature_value})
              </span>
            </div>

            <div className="shap-bar-track">
              <div
                className={`shap-bar-fill ${isRiskIncrease ? 'positive' : 'negative'}`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              {feat.interpretation}
            </div>
          </div>
        );
      })}

      <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', marginTop: '12px' }}>
        <strong>Model Causality Disclaimer:</strong> {disclaimer}
      </div>
    </div>
  );
}
