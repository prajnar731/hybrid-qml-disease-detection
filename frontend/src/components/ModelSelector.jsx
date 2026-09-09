import React from 'react';

export default function ModelSelector({ models, selectedModelId, onSelectModel }) {
  if (!models || models.length === 0) {
    return (
      <div className="model-selector-group">
        <label className="selector-label">Classification Model / VQC Architecture</label>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.84rem' }}>Loading models from API...</div>
      </div>
    );
  }

  return (
    <div className="model-selector-group">
      <label className="selector-label">Classification Model / Quantum Architecture</label>
      <div className="model-options-grid">
        {models.map((model) => {
          const isSelected = selectedModelId === model.model_id;
          const isQuantum = model.model_type === 'hybrid_quantum';

          return (
            <button
              key={model.model_id}
              type="button"
              className={`model-option-btn ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectModel(model.model_id)}
            >
              <div className="model-option-title">
                {isQuantum ? '⚛️ ' : '💻 '}
                {model.model_name}
              </div>
              <div className="model-option-specs">
                {isQuantum ? (
                  <span>
                    {model.qubits} Qubits • Depth {model.circuit_depth} • {model.trainable_parameters} Params
                  </span>
                ) : (
                  <span>{model.feature_representation}</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
