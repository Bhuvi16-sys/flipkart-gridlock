import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LeftPanelMap from './components/LeftPanelMap';
import RightPanel from './components/RightPanel';
import FooterMetrics from './components/FooterMetrics';
import Toast from './components/Toast';
import './index.css';

function App() {
  const [latency, setLatency] = useState(null);
  const [mapTarget, setMapTarget] = useState(null);

  const handleInferenceMetrics = (metrics) => {
    if (metrics.latency) {
      setLatency(metrics.latency);
    }
  };

  const handleMapTargetUpdate = (lat, lon) => {
    setMapTarget([lat, lon]);
  };

  return (
    <div className="app-container">
      <Navbar />
      <div className="main-content">
        <LeftPanelMap mapTarget={mapTarget} />
        <RightPanel 
          onInferenceMetrics={handleInferenceMetrics} 
          onMapTargetUpdate={handleMapTargetUpdate}
        />
      </div>
      <FooterMetrics latency={latency} />
      <Toast />
    </div>
  );
}

export default App;
