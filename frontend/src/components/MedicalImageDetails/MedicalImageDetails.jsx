import React, { useState, useEffect } from 'react';
import './MedicalImageDetails.css';

const MedicalImageDetails = ({ imageId, token }) => {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8001/medical-images/${imageId}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch image details');
        
        const data = await response.json();
        setDetails(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [imageId, token]);

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this analysis?')) return;
    
    try {
      setDeleting(true);
      const response = await fetch(`http://localhost:8001/medical-images/${imageId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to delete image');
      
      alert('Analysis deleted successfully');
      window.location.reload();
    } catch (err) {
      alert(err.message);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) return <div className="details-loading">Loading details...</div>;
  if (error) return <div className="details-error">Error: {error}</div>;
  if (!details) return null;

  return (
    <div className="image-details">
      <div className="details-section">
        <h3>Analysis Results</h3>
        <div className="results-grid">
          <div className="result-card">
            <div className="result-label">Scan Type</div>
            <div className="result-value">{details.scan_type}</div>
            <div className="result-confidence">
              Confidence: {Math.round(details.scan_type_confidence * 100)}%
            </div>
          </div>
          
          <div className="result-card">
            <div className="result-label">Anatomy</div>
            <div className="result-value">{details.anatomy}</div>
            <div className="result-confidence">
              Confidence: {Math.round(details.anatomy_confidence * 100)}%
            </div>
          </div>
          
          <div className="result-card">
            <div className="result-label">Diagnosis</div>
            <div className="result-value">{details.disease}</div>
            <div className="result-confidence">
              Confidence: {Math.round(details.disease_confidence * 100)}%
            </div>
          </div>
        </div>
      </div>
      
      <div className="details-section">
        <h3>Explanation</h3>
        <div className="explanation-text">
          {details.explanation || "No explanation available for this analysis."}
        </div>
      </div>
      
      {details.overlay_image_url && (
        <div className="details-section">
          <h3>Visual Analysis</h3>
          <div className="image-overlay">
            <img 
              src={details.overlay_image_url} 
              alt="Analysis overlay" 
              className="overlay-image"
            />
            <div className="overlay-caption">
              AI-generated visualization highlighting areas of concern
            </div>
          </div>
        </div>
      )}
      
      <div className="details-actions">
        <button 
          className="delete-button"
          onClick={handleDelete}
          disabled={deleting}
        >
          {deleting ? 'Deleting...' : 'Delete Analysis'}
        </button>
      </div>
    </div>
  );
};

export default MedicalImageDetails;