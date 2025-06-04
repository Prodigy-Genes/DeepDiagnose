import React, { useState, useEffect } from "react";
import ImageUpload from "../../components/ImageUpload/ImageUpload";
import PredictionResult from "../../components/PredictionResult/PredictionResult";
import "./UploadImage.css";

const UploadImage = () => {
  // State for prediction result
  const [result, setResult] = useState(null);
  // State for animation control
  const [showAnimation, setShowAnimation] = useState(true);

  // Callback to receive prediction data
  const handleResult = (data) => {
    setResult(data);
  };

  // Handle when upload begins
  const handleUploadStart = () => {
    setResult(null);
  };

  // Hide intro animation after it plays
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowAnimation(false);
    }, 3000);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    // Main upload page component 
    <div className="upload-page">
      {/* Intro animation */}
      {showAnimation && (
        <div className="scanner-animation">
          <div className="scan-line"></div>
          <div className="scanner-text">Welcome to deepdiagnose</div>
        </div>
      )}
      
      {/* Header section */}
      <header className="upload-header">
        <div className="header-content">
          <div className="logo-container">
            <h1>deep<span>diagnose</span></h1>
            <div className="logo-tagline">AI-powered x-ray images analysis</div>
          </div>
          
          {/* Navigation menu */}
          <div className="header-info">
            <div className="info-item">
              <i className="fas fa-microchip"></i>
              <span>AI-Powered Analysis</span>
            </div>
            <div className="info-item">
              <i className="fas fa-brain"></i>
              <span>Neural Networks</span>
            </div>
            <div className="info-item">
              <i className="fas fa-shield-alt"></i>
              <span>HIPAA Compliant</span>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main content area */}
      <main className="upload-main">
        <div className="content-container">
          <div className="upload-section">
            <div className="section-intro">
              <h2>X-Ray Image Analysis</h2>
              <p className="disclaimer">
                <strong>Disclaimer:</strong> This tool can only predict <b>Pneumonia</b>, <b>Covid-19</b> and <b>Osteoarthritis</b> from x-ray and CT medical scans.
              </p><br />
              <p>Upload your x-ray image to get AI-powered diagnosis assistance in seconds</p>
            </div>
            
            {/* Image upload component */}
            <ImageUpload onResult={handleResult} onUploadStart={handleUploadStart} />
          </div>
          
          {/* Prediction result appears once available */}
          {result && (
            <div className="result-section">
              <div className="result-header">
                <h2>Diagnostic Analysis</h2>
                <div className="divider"></div>
              </div>
              <PredictionResult result={result} />
            </div>
          )}
        </div>
      </main>
      
      <div className="tech-details">
        <div className="tech-item">
          <div className="tech-icon">AI</div>
          <div className="tech-info">
            <div className="tech-title">Deep Learning</div>
            <div className="tech-desc">Trained on 10,000+ labeled x-rays</div>
          </div>
        </div>
        <div className="tech-item">
          <div className="tech-icon">94%</div>
          <div className="tech-info">
            <div className="tech-title">Accuracy</div>
            <div className="tech-desc">On standard test datasets</div>
          </div>
        </div>
        
      </div>
      
      <footer className="upload-footer">
        <div className="footer-content">
          <div className="footer-logo">deepdiagnose</div>
          <div className="footer-disclaimer">
            For assistance purposes only. Not a replacement for professional medical diagnosis.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default UploadImage;