import React from 'react';

export default function Navbar({ backendOnline, onNavigate, activeSection }) {
  const navItems = [
    { id: 'overview', label: 'Overview' },
    { id: 'prediction', label: 'Screening' },
    { id: 'benchmark', label: 'Benchmarks' },
    { id: 'optimization', label: 'Quantum Resources' },
    { id: 'noise', label: 'Noise Analysis' },
  ];

  const handleNavClick = (id) => {
    onNavigate(id);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <span className="brand-icon">⚛️</span>
          <div>
            <div className="brand-title">Hybrid QML Disease Detection</div>
          </div>
          <span className="brand-badge">SIH26139</span>
        </div>

        <nav>
          <ul className="nav-links">
            {navItems.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className={`nav-link ${activeSection === item.id ? 'active' : ''}`}
                  onClick={() => handleNavClick(item.id)}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="backend-status-pill" title={backendOnline ? 'FastAPI API Connected' : 'FastAPI API Offline'}>
          <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`} />
          <span>{backendOnline ? 'API Connected' : 'API Offline (Port 8000)'}</span>
        </div>
      </div>
    </header>
  );
}
