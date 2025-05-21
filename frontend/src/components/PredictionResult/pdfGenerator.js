export const generateMedicalReport = (result) => {
  const {
    anatomy,
    anatomy_confidence,
    disease,
    disease_confidence,
    overlay_image,
    explanation,
  } = result;

  const currentDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  const reportHTML = `
    <!DOCTYPE html>
    <html>
      <head>
        <title>Medical Analysis Report - ${currentDate}</title>
        <meta charset="UTF-8">
        <style>
          @page {
            size: A4;
            margin: 0.5in;
          }
          
          * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
          }
          
          body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
          }
          
          .report-container {
            max-width: 100%;
            padding: 20px;
          }
          
          /* Header Styles */
          .report-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #0070f3;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 30px 20px 20px;
            border-radius: 8px;
          }
          
          .logo {
            font-size: 32px;
            font-weight: bold;
            color: #0070f3;
            margin-bottom: 5px;
          }
          
          .report-title {
            font-size: 24px;
            color: #1a202c;
            margin-bottom: 8px;
            font-weight: 600;
          }
          
          .report-subtitle {
            font-size: 14px;
            color: #4a5568;
            margin-bottom: 15px;
          }
          
          .report-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          }
          
          .meta-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #4a5568;
          }
          
          .meta-icon {
            width: 16px;
            height: 16px;
            fill: #0070f3;
          }
          
          /* Section Styles */
          .report-section {
            margin-bottom: 35px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            overflow: hidden;
          }
          
          .section-header {
            background: linear-gradient(135deg, #0070f3 0%, #0056b3 100%);
            color: white;
            padding: 15px 20px;
            font-size: 18px;
            font-weight: 600;
          }
          
          .section-content {
            padding: 25px;
          }
          
          /* Results Grid */
          .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
          }
          
          .result-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            background: #f8fafc;
          }
          
          .result-title {
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
          }
          
          .result-value {
            font-size: 20px;
            font-weight: bold;
            color: #1a202c;
            margin-bottom: 15px;
          }
          
          .confidence-container {
            margin-bottom: 10px;
          }
          
          .confidence-label {
            font-size: 14px;
            color: #4a5568;
            margin-bottom: 6px;
          }
          
          .confidence-bar-container {
            background: #e2e8f0;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
          }
          
          .confidence-bar {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            border-radius: 6px;
            transition: width 0.3s ease;
            position: relative;
          }
          
          .confidence-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
          }
          
          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
          
          .confidence-percentage {
            font-size: 14px;
            font-weight: 600;
            color: #059669;
            margin-top: 4px;
          }
          
          /* Explanation Section */
          .explanation-content {
            background: #f7fafc;
            border-left: 4px solid #0070f3;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            line-height: 1.8;
            color: #2d3748;
          }
          
          /* Image Section */
          .image-container {
            text-align: center;
            margin: 20px 0;
          }
          
          .analysis-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
          }
          
          .image-caption {
            font-size: 14px;
            color: #4a5568;
            margin-top: 10px;
            font-style: italic;
          }
          
          /* Footer */
          .report-footer {
            margin-top: 50px;
            padding: 25px;
            background: #f7fafc;
            border-radius: 8px;
            border-top: 3px solid #0070f3;
          }
          
          .disclaimer {
            background: #fef7e0;
            border: 1px solid #f6d55c;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
          }
          
          .disclaimer-title {
            font-weight: 600;
            color: #92400e;
            margin-bottom: 5px;
          }
          
          .disclaimer-text {
            font-size: 14px;
            color: #92400e;
            line-height: 1.6;
          }
          
          .copyright {
            text-align: center;
            font-size: 12px;
            color: #6b7280;
            margin-top: 15px;
          }
          
          /* Print Optimizations */
          @media print {
            body {
              font-size: 12px;
            }
            
            .report-container {
              padding: 0;
            }
            
            .section-header {
              -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
            }
            
            .confidence-bar {
              -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
            }
            
            .analysis-image {
              max-height: 300px;
            }
          }
          
          /* Utility Classes */
          .flex {
            display: flex;
          }
          
          .items-center {
            align-items: center;
          }
          
          .justify-between {
            justify-content: space-between;
          }
          
          .gap-2 {
            gap: 0.5rem;
          }
          
          .text-center {
            text-align: center;
          }
          
          .font-semibold {
            font-weight: 600;
          }
          
          .text-blue-600 {
            color: #0070f3;
          }
        </style>
      </head>
      <body>
        <div class="report-container">
          <!-- Header -->
          <header class="report-header">
            <div class="logo">deepdiagnose</div>
            <h1 class="report-title">AI Medical Analysis Report</h1>
            <p class="report-subtitle">Advanced x-ray Image Analysis</p>
            <div class="report-meta">
              <div class="meta-item">
                <svg class="meta-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.11 0-1.99.9-1.99 2L3 19c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11zM7 10h5v5H7z"/>
                </svg>
                <span>Generated: ${currentDate}</span>
              </div>
              <div class="meta-item">
                <svg class="meta-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                <span>AI Analysis Complete</span>
              </div>
              <div class="meta-item">
                <svg class="meta-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M9 11H7v6h2v-6zm4 0h-2v6h2v-6zm4 0h-2v6h2v-6zm2.5-9H19v2H5V2h3.5l1-1h5l1 1zm0 4H5v16h14V6z"/>
                </svg>
                <span>Report ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
              </div>
            </div>
          </header>

          <!-- Analysis Results Section -->
          <section class="report-section">
            <div class="section-header">
              📊 Analysis Results
            </div>
            <div class="section-content">
              <div class="results-grid">
                <!-- Anatomy Analysis -->
                <div class="result-card">
                  <h3 class="result-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                    </svg>
                    Anatomy Classification
                  </h3>
                  <div class="result-value">${anatomy}</div>
                  <div class="confidence-container">
                    <div class="confidence-label">Confidence Level</div>
                    <div class="confidence-bar-container">
                      <div class="confidence-bar" style="width: ${anatomy_confidence * 100}%"></div>
                    </div>
                    <div class="confidence-percentage">${(anatomy_confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <!-- Disease Detection -->
                <div class="result-card">
                  <h3 class="result-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                    Disease Detection
                  </h3>
                  <div class="result-value">${disease}</div>
                  <div class="confidence-container">
                    <div class="confidence-label">Confidence Level</div>
                    <div class="confidence-bar-container">
                      <div class="confidence-bar" style="width: ${disease_confidence * 100}%"></div>
                    </div>
                    <div class="confidence-percentage">${(disease_confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          ${explanation ? `
          <!-- Expert Analysis Section -->
          <section class="report-section">
            <div class="section-header">
              🔬 Expert Analysis
            </div>
            <div class="section-content">
              <div class="explanation-content">
                ${explanation}
              </div>
            </div>
          </section>
          ` : ''}

          ${overlay_image ? `
          <!-- Visual Analysis Section -->
          <section class="report-section">
            <div class="section-header">
              🖼️ Visual Analysis
            </div>
            <div class="section-content">
              <div class="image-container">
                <img src="${overlay_image}" alt="AI Analysis Overlay" class="analysis-image" />
                <div class="image-caption">AI-generated overlay showing areas of interest in the medical image</div>
              </div>
            </div>
          </section>
          ` : ''}

          <!-- Footer -->
          <footer class="report-footer">
            <div class="disclaimer">
              <div class="disclaimer-title">⚠️ Important Medical Disclaimer</div>
              <div class="disclaimer-text">
                This AI-generated analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. The AI system is a diagnostic aid and its results should be interpreted within the context of clinical findings and medical history.
              </div>
            </div>
            <div class="copyright">
              © ${new Date().getFullYear()} deepdiagnose. All rights reserved. | Generated using advanced machine learning algorithms.
            </div>
          </footer>
        </div>
      </body>
    </html>
  `;

  return reportHTML;
};

export const downloadPDFReport = (result) => {
  const reportHTML = generateMedicalReport(result);
  
  // Create a new window for printing
  const printWindow = window.open('', '_blank', 'width=800,height=600');
  
  if (printWindow) {
    printWindow.document.write(reportHTML);
    printWindow.document.close();
    
    // Wait for content to load, then trigger print dialog
    printWindow.onload = () => {
      setTimeout(() => {
        printWindow.print();
        printWindow.focus();
      }, 500);
    };
  } else {
    // Fallback: create blob and download
    const blob = new Blob([reportHTML], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medical-report-${new Date().toISOString().split('T')[0]}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
};