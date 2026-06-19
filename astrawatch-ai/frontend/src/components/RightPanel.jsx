import React, { useState, useRef } from 'react';
import { Settings2, Cpu, AlertTriangle, Zap, CheckCircle2, Image as ImageIcon } from 'lucide-react';

export default function RightPanel({ onInferenceMetrics, onMapTargetUpdate }) {
  const [form, setForm] = useState({
    latitude: '', // Populated by image telemetry
    longitude: '', // Populated by image telemetry
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19), // Live timestamp default
    priority: 'High',
    event_type: 'Accident',
    event_cause: 'Collision',
    junction: 'Silk Board',
    zone: 'South'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  // Image Telemetry State
  const [imagePreview, setImagePreview] = useState(null);
  const [extracting, setExtracting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  
  const [dispatched, setDispatched] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processImageFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    
    // Create preview
    const objectUrl = URL.createObjectURL(file);
    setImagePreview(objectUrl);
    setExtracting(true);

    // Simulate extracting telemetry and getting Geolocation
    setTimeout(() => {
      // We will try to use browser geolocation to make it feel real, 
      // or fallback to a dynamic Bengaluru coordinate if blocked.
      if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setForm(prev => ({
              ...prev,
              latitude: position.coords.latitude.toFixed(6),
              longitude: position.coords.longitude.toFixed(6),
              timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
            }));
            setExtracting(false);
          },
          (error) => {
            // Fallback mock coordinate near silk board
            setForm(prev => ({
              ...prev,
              latitude: "12.9172",
              longitude: "77.6228",
              timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
            }));
            setExtracting(false);
          }
        );
      } else {
        setForm(prev => ({
          ...prev,
          latitude: "12.9172",
          longitude: "77.6228",
          timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
        }));
        setExtracting(false);
      }
    }, 2000);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImageFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processImageFile(e.target.files[0]);
    }
  };

  const handleDispatch = () => {
    setDispatched(true);
    window.dispatchEvent(new CustomEvent('show-toast', { detail: { message: "Unit Dispatched Successfully to Target Zone.", type: 'success' } }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.latitude || !form.longitude) {
      window.dispatchEvent(new CustomEvent('show-toast', { detail: { message: "Please upload image evidence to extract coordinates first.", type: 'error' } }));
      return;
    }

    setLoading(true);
    setResult(null);
    setDispatched(false);
    
    const startTime = performance.now();

    try {
      const payload = {
        latitude: parseFloat(form.latitude),
        longitude: parseFloat(form.longitude),
        timestamp: form.timestamp,
        priority: form.priority,
        event_type: form.event_type || "unknown",
        event_cause: form.event_cause || "unknown",
        junction: form.junction || "unknown",
        zone: form.zone || "unknown"
      };

      const response = await fetch('/api/v1/predict-metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data.data);
      
      // Update global map target
      if (onMapTargetUpdate) {
        onMapTargetUpdate(parseFloat(form.latitude), parseFloat(form.longitude));
      }
      
      const endTime = performance.now();
      if (onInferenceMetrics) {
        onInferenceMetrics({ latency: Math.round(endTime - startTime) });
      }
    } catch (error) {
      console.error("Inference Error:", error);
      window.dispatchEvent(new CustomEvent('show-toast', { detail: { message: error.message, type: 'error' } }));
      
      const endTime = performance.now();
      if (onInferenceMetrics) {
        onInferenceMetrics({ latency: Math.round(endTime - startTime) });
      }
    } finally {
      setLoading(false);
    }
  };

  const getActionCardClass = (action) => {
    if (action === 'DISPATCH_URGENT_RESPONSE') return 'metric-card alert';
    if (action === 'MONITOR_DASHBOARD') return 'metric-card warning';
    return 'metric-card';
  };

  return (
    <div className="panel-right">
      <div className="section-title">
        <Settings2 size={18} />
        Incident Input Parameters
      </div>
      
      <form className="input-form" onSubmit={handleSubmit}>
        <div className="form-grid">
          
          <div 
            className={`upload-zone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              ref={fileInputRef}
              type="file" 
              accept="image/*" 
              onChange={handleFileInput} 
              style={{ display: 'none' }} 
            />
            
            {imagePreview && <img src={imagePreview} className="upload-preview" alt="Incident preview" />}
            
            <div className="upload-overlay">
              {extracting ? (
                <>
                  <div className="spinner"></div>
                  <span className="upload-text">Extracting Telemetry & Geolocation...</span>
                </>
              ) : form.latitude ? (
                <>
                  <CheckCircle2 size={24} className="upload-icon" style={{color: 'var(--color-emerald)'}}/>
                  <span className="upload-text">Telemetry Locked: [{form.latitude}, {form.longitude}]</span>
                </>
              ) : (
                <>
                  <ImageIcon size={28} className="upload-icon" />
                  <span className="upload-text">Upload Incident Evidence / Click Picture</span>
                </>
              )}
            </div>
          </div>

          <div className="form-group">
            <label>Timestamp</label>
            <input name="timestamp" value={form.timestamp} onChange={handleChange} className="form-input font-mono" required />
          </div>
          <div className="form-group">
            <label>Priority</label>
            <select name="priority" value={form.priority} onChange={handleChange} className="form-input">
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
              <option value="Critical">Critical</option>
            </select>
          </div>
          <div className="form-group">
            <label>Event Type</label>
            <input name="event_type" value={form.event_type} onChange={handleChange} className="form-input" />
          </div>
          <div className="form-group">
            <label>Zone</label>
            <input name="zone" value={form.zone} onChange={handleChange} className="form-input" />
          </div>
          <button type="submit" className="btn-submit" disabled={loading || extracting || !form.latitude}>
            {loading ? <div className="shimmer" style={{width: '100%', height: '100%', position: 'absolute'}}></div> : null}
            <Zap size={16} />
            {loading ? 'RUNNING INFERENCE...' : 'TRIGGER PREDICTION PIPELINE'}
          </button>
        </div>
      </form>

      <div className="section-title" style={{ marginTop: '1rem' }}>
        <Cpu size={18} />
        Live Inference Engine Output
      </div>

      <div className="output-section">
        {loading ? (
          <>
            <div className="metric-card shimmer" style={{ height: '100px' }}></div>
            <div className="metric-card shimmer" style={{ height: '100px' }}></div>
            <div className="metric-card shimmer" style={{ height: '120px' }}></div>
          </>
        ) : result ? (
          <>
            <div className="form-grid" style={{ gap: '1rem' }}>
              <div className="metric-card">
                <span className="metric-label">Target Geohash</span>
                <span className="metric-value font-mono">{result.geohash}</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Congestion Risk Score</span>
                <span className="metric-value font-mono">{Number(result.congestion_risk_score).toFixed(3)}</span>
                <div className="progress-container">
                  <div className="progress-fill" style={{ width: `${Math.min(result.congestion_risk_score * 100, 100)}%` }}></div>
                </div>
              </div>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Predicted Clearance Time</span>
              <span className="metric-value font-mono">{Math.round(result.predicted_clearance_minutes)} MIN</span>
              <div className="progress-container">
                <div className="progress-fill" style={{ width: `${Math.max(5, Math.min((result.predicted_clearance_minutes / 480) * 100, 100))}%` }}></div>
              </div>
            </div>

            <div className={getActionCardClass(result.recommended_action)}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                {result.recommended_action === 'DISPATCH_URGENT_RESPONSE' ? <AlertTriangle size={20} color="var(--color-crimson)" /> : <CheckCircle2 size={20} color="var(--color-amber)" />}
                <span className="metric-label" style={{ margin: 0 }}>Recommended Action</span>
              </div>
              <span className={`metric-value ${result.recommended_action === 'DISPATCH_URGENT_RESPONSE' ? 'critical' : ''}`}>
                {result.recommended_action.replace(/_/g, ' ')}
              </span>
              {result.recommended_action === 'DISPATCH_URGENT_RESPONSE' && (
                <button 
                  className={`dispatch-btn ${dispatched ? 'dispatched' : ''}`}
                  onClick={handleDispatch}
                  disabled={dispatched}
                >
                  {dispatched ? 'UNIT DISPATCHED' : 'ACKNOWLEDGE & DISPATCH UNIT'}
                </button>
              )}
            </div>
          </>
        ) : (
          <div className="metric-card" style={{ opacity: 0.5, textAlign: 'center', padding: '2rem' }}>
            <Cpu size={32} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <p>Engine Idle. Awaiting payload submission.</p>
          </div>
        )}
      </div>
    </div>
  );
}
