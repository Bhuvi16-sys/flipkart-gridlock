import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Clock } from 'lucide-react';

export default function Navbar() {
  const [time, setTime] = useState(new Date().toISOString().replace('T', ' ').substring(0, 19));

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toISOString().replace('T', ' ').substring(0, 19));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <Activity className="nav-logo-icon" size={24} />
        AstraWatch AI Engine v1.1
      </div>
      <div className="nav-status">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-mono)' }}>
          <Clock size={16} />
          {time} UTC
        </div>
        <div className="badge-healthy">
          <ShieldCheck size={16} />
          Connected / Healthy
          <div className="status-dot"></div>
        </div>
      </div>
    </nav>
  );
}
