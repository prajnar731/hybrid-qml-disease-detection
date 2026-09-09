import React from 'react';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div>
          <strong style={{ color: '#ffffff' }}>Hybrid Quantum Machine Learning Platform for Early Disease Detection</strong>
        </div>
        <div>
          Smart India Hackathon 2024 • Problem ID: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-cyan)' }}>SIH26139</span>
        </div>
        <div style={{ maxWidth: '850px', fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '8px' }}>
          <strong>Regulatory & Diagnostic Disclaimer:</strong> This system is an algorithmic and experimental engineering research prototype developed for vocal acoustic biomarker processing on the UCI Oxford Parkinson's dataset. It is <strong>NOT</strong> a medical diagnostic instrument and has not been cleared or approved by clinical regulatory authorities. Predictions and reliability metrics must not be used to direct patient care, diagnose medical conditions, or replace clinical consultations.
        </div>
      </div>
    </footer>
  );
}
