import React, { useRef } from "react";
import "./PredictionResult.css";
import { formatDate } from "./utils"; // Assume we have a utility function for date formatting

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
  
  // Generate PDF-like document as HTML for download
  const generatePDF = () => {
    
    // Create a new window/tab with the report content
    const printWindow = window.open('', '_blank');
    
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Medical Analysis Report - ${currentDate}</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
            }
            .report-header {
              text-align: center;
              margin-bottom: 30px;
              padding-bottom: 10px;
              border-bottom: 2px solid #2563eb;
            }
            .report-logo {
              font-size: 24px;
              font-weight: bold;
              color: #2563eb;
            }
            .report-title {
              font-size: 22px;
              margin: 10px 0;
            }
            .report-date {
              font-size: 14px;
              color: #666;
            }
            .report-section {
              margin-bottom: 20px;
            }
            .section-title {
              font-size: 18px;
              font-weight: bold;
              margin-bottom: 10px;
              color: #2563eb;
            }
            .data-row {
              display: flex;
              margin-bottom: 10px;
            }
            .data-label {
              font-weight: bold;
              width: 150px;
            }
            .confidence-bar {
              height: 10px;
              background-color: #e5e7eb;
              border-radius: 5px;
              margin-top: 5px;
              overflow: hidden;
            }
            .confidence-fill {
              height: 100%;
              background-color: #2563eb;
            }
            .explanation {
              background-color: #f3f4f6;
              padding: 15px;
              border-radius: 5px;
              font-style: italic;
            }
            .report-footer {
              margin-top: 50px;
              text-align: center;
              font-size: 12px;
              color: #666;
            }
            img {
              max-width: 100%;
              border-radius: 5px;
            }
          </style>
        </head>
        <body>
          <div class="report-header">
            <div class="report-logo">DeepDiagnose</div>
            <h1 class="report-title">Medical Image Analysis Report</h1>
            <div class="report-date">Generated on ${currentDate}</div>
          </div>

          <div class="report-section">
            <div class="section-title">Analysis Results</div>
            
            <div class="data-row">
              <div class="data-label">Anatomy:</div>
              <div>${anatomy}</div>
            </div>
            
            <div class="data-row">
              <div class="data-label">Confidence:</div>
              <div>
                ${(anatomy_confidence * 100).toFixed(2)}%
                <div class="confidence-bar">
                  <div class="confidence-fill" style="width: ${anatomy_confidence * 100}%"></div>
                </div>
              </div>
            </div>
            
            <div class="data-row">
              <div class="data-label">Identified Disease:</div>
              <div>${disease}</div>
            </div>
            
            <div class="data-row">
              <div class="data-label">Disease Confidence:</div>
              <div>
                ${(disease_confidence * 100).toFixed(2)}%
                <div class="confidence-bar">
                  <div class="confidence-fill" style="width: ${disease_confidence * 100}%"></div>
                </div>
              </div>
            </div>
          </div>

          ${explanation ? `
          <div class="report-section">
            <div class="section-title">Expert Explanation</div>
            <div class="explanation">${explanation}</div>
          </div>
          ` : ''}

          ${overlay_image ? `
          <div class="report-section">
            <div class="section-title">Visual Analysis</div>
            <img src="${overlay_image}" alt="AI Overlay Analysis" />
          </div>
          ` : ''}

          <div class="report-footer">
            <p>This is an AI-generated report and should be reviewed by a healthcare professional.</p>
            <p>© ${new Date().getFullYear()} MedScan AI System</p>
          </div>
        </body>
      </html>
    `);
    
    printWindow.document.close();
    
    // Give the browser a moment to process the document before printing
    setTimeout(() => {
      printWindow.print();
    }, 250);
  };

  // Handle direct download as PDF
  const handleDownload = () => {
    generatePDF();
  };

  return (
    <div className="prediction-container">
      <div className="prediction-card" ref={reportRef}>
        <div className="prediction-header">
          <h2 className="prediction-title">Analysis Results</h2>
          <span className="prediction-date">{currentDate}</span>
        </div>
        
        <div className="prediction-content">
          <div className="result-section">
            <h3 className="section-title">Diagnostic Information</h3>
            
            <div className="result-item">
              <div className="result-label">Anatomy</div>
              <div className="result-value">{anatomy}</div>
              <div className="confidence-wrapper">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${anatomy_confidence * 100}%` }}
                  ></div>
                </div>
                <span className="confidence-text">
                  {(anatomy_confidence * 100).toFixed(2)}% confidence
                </span>
              </div>
            </div>
            
            <div className="result-item">
              <div className="result-label">Disease</div>
              <div className="result-value">{disease}</div>
              <div className="confidence-wrapper">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${disease_confidence * 100}%` }}
                  ></div>
                </div>
                <span className="confidence-text">
                  {(disease_confidence * 100).toFixed(2)}% confidence
                </span>
              </div>
            </div>
          </div>
          
          {explanation && (
            <div className="result-section">
              <h3 className="section-title">Expert Explanation</h3>
              <div className="explanation-box">
                {explanation}
              </div>
            </div>
          )}
          
          {overlay_image && (
            <div className="result-section">
              <h3 className="section-title">Visual Analysis</h3>
              <div className="image-container">
                <img
                  src={overlay_image}
                  alt="Model overlay visualization"
                  className="overlay-image"
                />
              </div>
            </div>
          )}
          
          <div className="disclaimer">
            This is an AI-generated analysis and should be reviewed by a healthcare professional.
          </div>
        </div>
      </div>
      
      <div className="action-buttons">
        <button 
          className="download-button"
          onClick={handleDownload}
        >
          <svg 
            className="button-icon" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          Download PDF Report
        </button>
      </div>
    </div>
  );
};

export default PredictionResult;