import React, { useState, useEffect } from "react";
import ImageUpload from "../../components/ImageUpload/ImageUpload";
import PredictionResult from "../../components/PredictionResult/PredictionResult";
import "./UploadImage.css";
import UserStatusIndicator from "../../auth/UserStatusIndicator/userStatus_Indicator";
import SignIn from "../../auth/Sign-In/SignIn";
import { Link } from 'react-router-dom';


const UploadImage = () => {
  const [result, setResult] = useState(null);
  const [showAnimation, setShowAnimation] = useState(true);
  const [showSignIn, setShowSignIn] = useState(false);

  const handleShowSignIn = () => {
    console.log('🔓 Opening SignIn modal from UploadImage');
    setShowSignIn(true);
  };
  
  const handleCloseSignIn = () => {
    console.log('🔒 Closing SignIn modal');
    setShowSignIn(false);
  };


  const handleResult = (data) => {
    setResult(data);
  };

  const handleUploadStart = () => {
    setResult(null);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowAnimation(false);
    }, 3000);
    
    return () => clearTimeout(timer);
  }, []);

  // Listen for custom login events as a backup
  useEffect(() => {
    const handleLoginRequest = (event) => {
      console.log('🔓 Login requested via custom event:', event.detail);
      handleShowSignIn();
    };

    document.addEventListener('requestLogin', handleLoginRequest);
    
    return () => {
      document.removeEventListener('requestLogin', handleLoginRequest);
    };
  }, []);

  return (
    <div className="upload-page">
      {/* SignIn Modal - Moved to root level */}
      {showSignIn && (
        <SignIn 
          onToggleAuth={() => {/* handle toggle to signup */}}
          onClose={handleCloseSignIn}
        />
      )}
      
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

          {/* User status indicator */}
          <UserStatusIndicator onLoginClick={handleShowSignIn} />
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
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4">
          <Link 
            to="/debug"
            className="bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-gray-700 transition"
          >
            🛠️ Debug Tools
          </Link>
        </div>
      )}
    </div>
  );
};

export default UploadImage;