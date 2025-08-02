import React, { useState, useEffect } from 'react';
import './MedicalStatistics.css';

const MedicalStatistics = ({ token }) => {
  const [stats, setStats] = useState({
    total_analyses: 0,
    successful_analyses: 0,
    failed_analyses: 0,
    disease_distribution: [], // Initialize as empty array
    average_confidence: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8001/medical-statistics', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch statistics');
        
        const data = await response.json();
        
        // Ensure disease_distribution is always an array
        const processedData = {
          ...data,
          disease_distribution: Array.isArray(data.disease_distribution) 
            ? data.disease_distribution 
            : []
        };
        
        setStats(processedData);
      } catch (err) {
        setError(err.message);
        console.error('Statistics fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    if (token) {
      fetchStats();
    }
  }, [token]);

  if (loading) return <div className="stats-loading">Loading statistics...</div>;
  if (error) return <div className="stats-error">Error: {error}</div>;
  if (!stats) return null;

  return (
    <div className="statistics-container">
      <h3>Medical Analysis Statistics</h3>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.total_analyses || 0}</div>
          <div className="stat-label">Total Analyses</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.successful_analyses || 0}</div>
          <div className="stat-label">Successful</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.failed_analyses || 0}</div>
          <div className="stat-label">Failed</div>
        </div>
      </div>
      
      <div className="disease-stats">
        <h4>Analysis by Condition</h4>
        <div className="disease-chart">
          {stats.disease_distribution && stats.disease_distribution.length > 0 ? (
            stats.disease_distribution.map((disease, index) => (
              <div 
                key={index} 
                className="disease-bar"
                style={{ width: `${disease.percentage || 0}%` }}
              >
                <div className="disease-label">{disease.disease || 'Unknown'}</div>
                <div className="disease-count">{disease.count || 0}</div>
              </div>
            ))
          ) : (
            <div className="no-disease-data">No disease distribution data available</div>
          )}
        </div>
      </div>
      
      <div className="confidence-stats">
        <h4>Average Confidence</h4>
        <div className="confidence-meter">
          <div 
            className="confidence-fill"
            style={{ width: `${stats.average_confidence || 0}%` }}
          >
            <span>{Math.round(stats.average_confidence || 0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MedicalStatistics;