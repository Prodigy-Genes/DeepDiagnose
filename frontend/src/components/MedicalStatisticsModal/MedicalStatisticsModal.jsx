import React, { useState, useEffect, useCallback } from 'react';
import './MedicalStatisticsModal.css';

const MedicalStatisticsModal = ({ isOpen, onClose, token }) => {
  const [stats, setStats] = useState({
    total_analyses: 0,
    successful_analyses: 0,
    failed_analyses: 0,
    disease_distribution: [],
    average_confidence: 0
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
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
  }, [token]);

  useEffect(() => {
    if (isOpen && token) {
      fetchStats();
    }
  }, [isOpen, token, fetchStats]);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content medical-stats-modal">
        <div className="modal-header">
          <h2>
            <i className="fas fa-chart-bar"></i>
            Medical Analysis Statistics
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            <i className="fas fa-times"></i>
          </button>
        </div>
        
        <div className="modal-body">
          {loading ? (
            <div className="modal-loading">
              <i className="fas fa-spinner fa-spin"></i>
              <span>Loading statistics...</span>
            </div>
          ) : error ? (
            <div className="modal-error">
              <i className="fas fa-exclamation-triangle"></i>
              <span>Error: {error}</span>
              <button onClick={fetchStats} className="retry-btn">
                <i className="fas fa-redo"></i>
                Retry
              </button>
            </div>
          ) : (
            <div className="stats-content">
              {/* Overview Cards */}
              <div className="stats-overview">
                <div className="stat-card primary">
                  <div className="stat-icon">
                    <i className="fas fa-clipboard-list"></i>
                  </div>
                  <div className="stat-info">
                    <div className="stat-value">{stats.total_analyses || 0}</div>
                    <div className="stat-label">Total Analyses</div>
                  </div>
                </div>
                
                <div className="stat-card success">
                  <div className="stat-icon">
                    <i className="fas fa-check-circle"></i>
                  </div>
                  <div className="stat-info">
                    <div className="stat-value">{stats.successful_analyses || 0}</div>
                    <div className="stat-label">Successful</div>
                  </div>
                </div>
                
                <div className="stat-card warning">
                  <div className="stat-icon">
                    <i className="fas fa-exclamation-circle"></i>
                  </div>
                  <div className="stat-info">
                    <div className="stat-value">{stats.failed_analyses || 0}</div>
                    <div className="stat-label">Failed</div>
                  </div>
                </div>
                
                <div className="stat-card info">
                  <div className="stat-icon">
                    <i className="fas fa-percentage"></i>
                  </div>
                  <div className="stat-info">
                    <div className="stat-value">{Math.round(stats.average_confidence || 0)}%</div>
                    <div className="stat-label">Avg Confidence</div>
                  </div>
                </div>
              </div>

              {/* Disease Distribution */}
              <div className="disease-analysis-section">
                <h3>
                  <i className="fas fa-pie-chart"></i>
                  Analysis by Condition
                </h3>
                
                {stats.disease_distribution && stats.disease_distribution.length > 0 ? (
                  <div className="disease-chart">
                    {stats.disease_distribution.map((disease, index) => (
                      <div key={index} className="disease-item">
                        <div className="disease-info">
                          <div className="disease-name">{disease.disease || 'Unknown'}</div>
                          <div className="disease-stats">
                            <span className="disease-count">{disease.count || 0} cases</span>
                            <span className="disease-percentage">{disease.percentage || 0}%</span>
                          </div>
                        </div>
                        <div className="disease-bar-container">
                          <div 
                            className="disease-bar"
                            style={{ 
                              width: `${Math.min(disease.percentage || 0, 100)}%`,
                              backgroundColor: `hsl(${(index * 137.5) % 360}, 70%, 50%)`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-data">
                    <i className="fas fa-chart-bar"></i>
                    <span>No disease distribution data available</span>
                  </div>
                )}
              </div>

              {/* Confidence Analysis */}
              <div className="confidence-section">
                <h3>
                  <i className="fas fa-tachometer-alt"></i>
                  Confidence Analysis
                </h3>
                <div className="confidence-meter-container">
                  <div className="confidence-meter">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${Math.min(stats.average_confidence || 0, 100)}%` }}
                    >
                      <span className="confidence-text">
                        {Math.round(stats.average_confidence || 0)}%
                      </span>
                    </div>
                  </div>
                  <div className="confidence-labels">
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="modal-footer">
          <button onClick={onClose} className="close-modal-btn">
            Close
          </button>
          <button onClick={fetchStats} className="refresh-btn">
            <i className="fas fa-sync-alt"></i>
            Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
};

export default MedicalStatisticsModal;