import React, { useState } from 'react';
import './MedicalHistoryItem.css';
import MedicalImageDetails from '../MedicalImageDetails/MedicalImageDetails';

const MedicalHistoryItem = ({ item, token }) => {
  const [showDetails, setShowDetails] = useState(false);
  
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConditionIcon = () => {
    switch(item.disease) {
      case 'Pneumonia': return 'fas fa-lungs';
      case 'Osteoarthritis': return 'fas fa-bone';
      case 'COVID-19': return 'fas fa-virus';
      default: return 'fas fa-file-medical';
    }
  };

  return (
    <>
      <div className="history-item" onClick={() => setShowDetails(!showDetails)}>
        <div className="item-icon">
          <i className={getConditionIcon()}></i>
        </div>
        
        <div className="item-content">
          <div className="item-header">
            <h3>{item.original_filename}</h3>
            <span className={`confidence-badge ${item.disease_confidence > 0.8 ? 'high' : 'medium'}`}>
              {Math.round(item.disease_confidence * 100)}%
            </span>
          </div>
          
          <div className="item-meta">
            <div className="meta-item">
              <i className="fas fa-stethoscope"></i>
              <span>{item.scan_type} • {item.anatomy}</span>
            </div>
            <div className="meta-item">
              <i className="fas fa-disease"></i>
              <span>{item.disease}</span>
            </div>
            <div className="meta-item">
              <i className="fas fa-clock"></i>
              <span>{formatDate(item.uploaded_at)}</span>
            </div>
          </div>
        </div>
        
        <div className="item-arrow">
          <i className={`fas fa-chevron-${showDetails ? 'up' : 'down'}`}></i>
        </div>
      </div>
      
      {showDetails && <MedicalImageDetails imageId={item.image_id} token={token} />}
    </>
  );
};

export default MedicalHistoryItem;