import React, { useState, useEffect } from 'react';
import './MedicalStatistics.css';

const MedicalStatistics = ({ token }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/medical-statistics', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch statistics');
        
        const data = await response.json();
        setStats(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [token]);

  if (loading) return <div className="stats-loading">Loading statistics...</div>;
  if (error) return <div className="stats-error">Error: {error}</div>;
  if (!stats) return null;

  return (
    <div className="statistics-container">
      <h3>Medical Analysis Statistics</h3>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.total_analyses}</div>
          <div className="stat-label">Total Analyses</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.successful_analyses}</div>
          <div className="stat-label">Successful</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.failed_analyses}</div>
          <div className="stat-label">Failed</div>
        </div>
      </div>
      
      <div className="disease-stats">
        <h4>Analysis by Condition</h4>
        <div className="disease-chart">
          {stats.disease_distribution.map((disease, index) => (
            <div 
              key={index} 
              className="disease-bar"
              style={{ width: `${disease.percentage}%` }}
            >
              <div className="disease-label">{disease.disease}</div>
              <div className="disease-count">{disease.count}</div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="confidence-stats">
        <h4>Average Confidence</h4>
        <div className="confidence-meter">
          <div 
            className="confidence-fill"
            style={{ width: `${stats.average_confidence}%` }}
          >
            <span>{Math.round(stats.average_confidence)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MedicalStatistics;