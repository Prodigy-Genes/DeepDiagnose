import React, { useRef } from "react";
import { Download, Calendar, Target, Activity, FileText, Heart } from 'lucide-react';
import "./PredictionResult.css";
import { formatDate } from "./utils"; // Assume we have a utility function for date formatting4
import { downloadPDFReport } from "./pdfGenerator";

const CircularProgress = ({ percentage, size = 120, strokeWidth = 8, label = "Confidence" }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="circular-progress" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="progress-svg">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="rgba(0, 208, 255, 0.2)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="var(--accent-blue)"
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="progress-circle"
        />
      </svg>
      <div className="progress-text">
        <span className="progress-percentage">{percentage.toFixed(1)}%</span>
        <span className="progress-label">{label}</span>
      </div>
    </div>
  );
};

const PredictionResult = ({ result }) => {
  // Reference for the content to be downloaded
  const reportRef = useRef(null);
  
  // Wait until we have a result object
  if (!result) return null;

  const {
    anatomy,
    anatomy_confidence,
    disease,
    disease_confidence,
    overlay_image,
    explanation,
  } = result;

  // Format date for the report
  const currentDate = formatDate(new Date());
  
  // Handle direct download as PDF using the new generator
  const handleDownload = () => {
    downloadPDFReport(result);
  };

  return (
    <div className="prediction-page">
      {/* Header */}
      <header className="prediction-header">
        <div className="header-content">
          <div className="logo-container">
            <h1>deep<span>diagnose</span></h1>
            <div className="logo-tagline">AI-Powered Medical Analysis</div>
          </div>
          <div className="header-info">
            <div className="info-item">
              <Calendar className="info-icon" size={16} />
              <span>{currentDate}</span>
            </div>
            <div className="info-item">
              <Activity className="info-icon" size={16} />
              <span>Analysis Complete</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="prediction-main">
        <div className="content-container" ref={reportRef}>
          
          {/* Results Header */}
          <div className="section-intro">
            <h2>Analysis Results</h2>
            <p>Advanced AI analysis of your medical data</p>
          </div>

          {/* Prediction Card */}
          <div className="prediction-card">
            
            {/* Anatomy Section */}
            <div className="diagnosis-section">
              <div className="diagnosis-content">
                <div className="diagnosis-header">
                  <Heart size={24} className="section-icon" />
                  <h3>Anatomy Classification</h3>
                </div>
                <div className="diagnosis-result">
                  {anatomy}
                </div>
              </div>
              
              <div className="confidence-section">
                <CircularProgress 
                  percentage={anatomy_confidence * 100} 
                  label="Anatomy Confidence"
                />
              </div>
            </div>

            {/* Disease Section */}
            <div className="diagnosis-section">
              <div className="diagnosis-content">
                <div className="diagnosis-header">
                  <Target size={24} className="section-icon" />
                  <h3>Disease Detection</h3>
                </div>
                <div className="diagnosis-result">
                  {disease}
                </div>
              </div>
              
              <div className="confidence-section">
                <CircularProgress 
                  percentage={disease_confidence * 100} 
                  label="Disease Confidence"
                />
              </div>
            </div>

            {/* Analysis Explanation */}
            {explanation && (
              <div className="explanation-section">
                <h4>Expert Analysis</h4>
                <div className="explanation-content">
                  {explanation}
                </div>
              </div>
            )}

            {/* Medical Image Visualization */}
            {overlay_image && (
              <div className="visualization-section">
                <h4>Visual Analysis</h4>
                <div className="image-container">
                  <img 
                    src={overlay_image} 
                    alt="AI Model overlay visualization"
                    className="overlay-image"
                  />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="action-section">
              <button 
                onClick={handleDownload}
                className="download-button"
              >
                <Download size={18} />
                <span>Download PDF Report</span>
              </button>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="disclaimer">
            <h1>
              This is an AI-generated analysis and should be reviewed by a healthcare professional. 
              Always consult with a qualified healthcare provider for medical decisions.
            </h1>  
          </div>
        </div>
      </main>
    </div>
  );
};

export default PredictionResult;