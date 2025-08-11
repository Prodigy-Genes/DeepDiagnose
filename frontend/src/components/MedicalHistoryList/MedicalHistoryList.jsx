import React, { useState, useEffect } from 'react';
import './MedicalHistoryList.css';
import MedicalHistoryItem from '../MedicalHistoryItem/MedicalHistoryItem';

const MedicalHistoryList = ({ token }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    disease: '',
    limit: 10,
    offset: 0
  });

  useEffect(() => {
    const fetchMedicalHistory = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8001/medical-images', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch medical history');
        
        const data = await response.json();
        setHistory(data.images);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMedicalHistory();
  }, [token, filters]);

  const handleFilterChange = (e) => {
    setFilters({ ...filters, [e.target.name]: e.target.value });
  };

  if (loading) return <div className="medical-history-loading">Loading history...</div>;
  if (error) return <div className="medical-history-error">Error: {error}</div>;

  return (
    <div className="medical-history-container">
      <div className="history-header">
        <h2>Medical Analysis History</h2>
        <div className="history-filters">
          <select name="disease" value={filters.disease} onChange={handleFilterChange}>
            <option value="">All Conditions</option>
            <option value="Pneumonia">Pneumonia</option>
            <option value="Osteoarthritis">Osteoarthritis</option>
            <option value="COVID-19">COVID-19</option>
          </select>
        </div>
      </div>
      
      <div className="history-list">
        {history.length > 0 ? (
          history.map(item => (
            <MedicalHistoryItem key={item.image_id} item={item} token={token} />
          ))
        ) : (
          <div className="no-history">No medical analysis history found</div>
        )}
      </div>
    </div>
  );
};

export default MedicalHistoryList;