import React from 'react';
import { Database, Activity, Code2 } from 'lucide-react';

export default function FooterMetrics({ latency }) {
  return (
    <footer className="footer-metrics">
      <div className="metric-item">
        <Activity size={14} />
        <span>Inference Latency: <span className="highlight">{latency || '--'}ms</span></span>
      </div>
      <div className="metric-item">
        <Database size={14} />
        <span>Bootstrap RAM Status: <span className="highlight">Cached</span></span>
      </div>
      <div className="metric-item">
        <Code2 size={14} />
        <span>Model Architecture: <span className="highlight">XGBoost Spatiotemporal</span></span>
      </div>
    </footer>
  );
}
