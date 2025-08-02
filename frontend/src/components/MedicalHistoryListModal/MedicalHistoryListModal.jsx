import React, { useState, useEffect, useCallback } from 'react';
import MedicalImageDetails from '../MedicalImageDetails/MedicalImageDetails';
import './MedicalHistoryListModal.css';

const MedicalHistoryModal = ({ isOpen, onClose, token }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedItem, setSelectedItem] = useState(null);
  const [filters, setFilters] = useState({
    disease: '',
    limit: 20,
    offset: 0
  });

  const fetchMedicalHistory = useCallback(async () => {
    if (!token) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams();
      if (filters.disease) queryParams.append('disease', filters.disease);
      queryParams.append('limit', filters.limit);
      queryParams.append('offset', filters.offset);
      
      const response = await fetch(`http://localhost:8001/medical-images?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch medical history');
      
      const data = await response.json();
      setHistory(data.images || []);
    } catch (err) {
      setError(err.message);
      console.error('Medical history fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [token, filters.disease, filters.limit, filters.offset]);

  useEffect(() => {
    if (isOpen && token) {
      fetchMedicalHistory();
    }
  }, [isOpen, token, fetchMedicalHistory]);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      if (selectedItem) {
        setSelectedItem(null);
      } else {
        onClose();
      }
    }
  };

  const handleFilterChange = (e) => {
    setFilters({ 
      ...filters, 
      [e.target.name]: e.target.value,
      offset: 0 // Reset offset when filtering
    });
  };

  const handleItemClick = (item) => {
    setSelectedItem(item);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content medical-history-modal">
        <div className="modal-header">
          <h2>
            <i className="fas fa-history"></i>
            Medical Analysis History
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            <i className="fas fa-times"></i>
          </button>
        </div>
        
        <div className="modal-body">
          {selectedItem ? (
            <div className="history-detail-view">
              <div className="detail-header">
                <button 
                  className="back-btn"
                  onClick={() => setSelectedItem(null)}
                >
                  <i className="fas fa-arrow-left"></i>
                  Back to History
                </button>
                <h3>Analysis Details</h3>
              </div>
              <MedicalImageDetails 
                imageId={selectedItem.image_id} 
                token={token} 
              />
            </div>
          ) : (
            <>
              {/* Filters */}
              <div className="history-filters">
                <div className="filter-group">
                  <label>Filter by condition:</label>
                  <select 
                    name="disease" 
                    value={filters.disease} 
                    onChange={handleFilterChange}
                  >
                    <option value="">All Conditions</option>
                    <option value="Pneumonia">Pneumonia</option>
                    <option value="Osteoarthritis">Osteoarthritis</option>
                    <option value="COVID-19">COVID-19</option>
                    <option value="Normal">Normal</option>
                  </select>
                </div>
                
                <div className="filter-group">
                  <label>Results per page:</label>
                  <select 
                    name="limit" 
                    value={filters.limit} 
                    onChange={handleFilterChange}
                  >
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                  </select>
                </div>
              </div>

              {/* History List */}
              {loading ? (
                <div className="modal-loading">
                  <i className="fas fa-spinner fa-spin"></i>
                  <span>Loading medical history...</span>
                </div>
              ) : error ? (
                <div className="modal-error">
                  <i className="fas fa-exclamation-triangle"></i>
                  <span>Error: {error}</span>
                  <button onClick={fetchMedicalHistory} className="retry-btn">
                    <i className="fas fa-redo"></i>
                    Retry
                  </button>
                </div>
              ) : history.length > 0 ? (
                <div className="history-grid">
                  {history.map(item => (
                    <div 
                      key={item.image_id} 
                      className="history-card"
                      onClick={() => handleItemClick(item)}
                    >
                      <div className="card-header">
                        <div className="analysis-type">
                          <i className="fas fa-x-ray"></i>
                          <span>{item.scan_type || 'Unknown Scan'}</span>
                        </div>
                        <div className="analysis-date">
                          {formatDate(item.created_at)}
                        </div>
                      </div>
                      
                      <div className="card-body">
                        <div className="diagnosis-info">
                          <div className="diagnosis">
                            <strong>{item.disease || 'No diagnosis'}</strong>
                          </div>
                          <div className="anatomy">
                            <i className="fas fa-search-location"></i>
                            {item.anatomy || 'Unknown anatomy'}
                          </div>
                        </div>
                        
                        <div className="confidence-info">
                          <div className={`confidence-badge ${getConfidenceColor(item.disease_confidence)}`}>
                            {Math.round((item.disease_confidence || 0) * 100)}% confidence
                          </div>
                        </div>
                      </div>
                      
                      <div className="card-footer">
                        <span className="view-details">
                          Click to view details
                          <i className="fas fa-arrow-right"></i>
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-data">
                  <i className="fas fa-folder-open"></i>
                  <span>No medical analysis history found</span>
                  {filters.disease && (
                    <button 
                      onClick={() => setFilters({...filters, disease: ''})}
                      className="clear-filter-btn"
                    >
                      Clear filter
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        
        {!selectedItem && (
          <div className="modal-footer">
            <button onClick={onClose} className="close-modal-btn">
              Close
            </button>
            <button onClick={fetchMedicalHistory} className="refresh-btn">
              <i className="fas fa-sync-alt"></i>
              Refresh
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MedicalHistoryModal;