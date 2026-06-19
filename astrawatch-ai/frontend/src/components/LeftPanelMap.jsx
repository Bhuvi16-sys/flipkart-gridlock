import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, useMap, Tooltip } from 'react-leaflet';
import { Map, Layers } from 'lucide-react';

// Component to handle dynamic panning
function MapController({ targetPos }) {
  const map = useMap();
  useEffect(() => {
    if (targetPos) {
      map.flyTo(targetPos, 14, {
        duration: 1.5,
        easeLinearity: 0.25
      });
    }
  }, [targetPos, map]);
  return null;
}

export default function LeftPanelMap({ mapTarget }) {
  const [showHotspots, setShowHotspots] = useState(true);
  const [showDispatches, setShowDispatches] = useState(true);

  // Bengaluru center
  const center = [12.9716, 77.5946];

  // Mock hotspots (Dense areas)
  const hotspots = [
    { pos: [12.9172, 77.6228], intensity: 0.8 }, // Silk Board
    { pos: [12.9304, 77.6784], intensity: 0.6 }, // Bellandur
    { pos: [12.9591, 77.6974], intensity: 0.9 }, // Marathahalli
    { pos: [13.0280, 77.5891], intensity: 0.5 }, // Hebbal
  ];

  return (
    <div className="panel-left">
      <div className="map-floating-controls">
        <div style={{ fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={14} /> Overlay Controls
        </div>
        <label className="control-toggle">
          <input 
            type="checkbox" 
            checked={showHotspots} 
            onChange={(e) => setShowHotspots(e.target.checked)} 
          />
          Toggle Gridlock Hotspots
        </label>
        <label className="control-toggle">
          <input 
            type="checkbox" 
            checked={showDispatches} 
            onChange={(e) => setShowDispatches(e.target.checked)} 
          />
          Show Active Dispatches
        </label>
      </div>

      <MapContainer 
        center={center} 
        zoom={12} 
        zoomControl={false}
        scrollWheelZoom={true} 
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          className="dark-tiles"
        />

        {showHotspots && hotspots.map((spot, i) => (
          <CircleMarker
            key={i}
            center={spot.pos}
            radius={spot.intensity * 30}
            pathOptions={{
              color: 'transparent',
              fillColor: 'var(--color-crimson)',
              fillOpacity: spot.intensity * 0.5,
            }}
          >
            <Tooltip direction="top" opacity={1}>
              <span style={{fontFamily: 'var(--font-mono)'}}>Risk Density: {spot.intensity}</span>
            </Tooltip>
          </CircleMarker>
        ))}

        {showDispatches && mapTarget && (
          <CircleMarker
            center={mapTarget}
            radius={15}
            pathOptions={{
              color: 'var(--color-crimson)',
              fillColor: 'var(--color-crimson)',
              fillOpacity: 1,
              className: 'sonar-pulse-icon'
            }}
          >
            <Tooltip permanent direction="bottom" offset={[0, 10]}>
              <span style={{fontWeight: 'bold', color: 'var(--color-crimson)'}}>ACTIVE DISPATCH</span>
            </Tooltip>
          </CircleMarker>
        )}

        <MapController targetPos={mapTarget} />
      </MapContainer>
    </div>
  );
}
