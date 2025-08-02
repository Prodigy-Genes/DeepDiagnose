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
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
          :root {
            --primary-dark: rgb(10, 31, 68);
            --primary-light: rgb(15, 44, 89);
            --accent-blue: rgb(0, 208, 255);
            --accent-blue-dark: rgb(0, 136, 255);
            --text-light: #ffffff;
            --text-dark: #333333;
            --grid-color: rgba(0, 208, 255, 0.15);
            --shadow-color: rgba(0, 0, 0, 0.2);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
          }

          @page {
            size: A4;
            margin: 0.5in;
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-light) 100%);
          }
          
          * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
          }
          
          body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            color: var(--text-light);
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-light) 100%);
            min-height: 100vh;
            position: relative;
          }

          body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
              linear-gradient(90deg, var(--grid-color) 1px, transparent 1px) 0 0 / 20px 20px,
              linear-gradient(0deg, var(--grid-color) 1px, transparent 1px) 0 0 / 20px 20px;
            opacity: 0.3;
            z-index: -1;
            pointer-events: none;
          }
          
          .report-container {
            max-width: 100%;
            padding: 20px;
            position: relative;
            z-index: 1;
          }
          
          /* Header Styles */
          .report-header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 20px 20px;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
          }

          .report-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%);
            z-index: 1;
          }
          
          .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 32px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 8px;
            letter-spacing: 2px;
            text-shadow: 0 0 10px rgba(0, 208, 255, 0.5);
            filter: drop-shadow(0 0 5px rgba(0, 208, 255, 0.3));
          }
          
          .report-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 24px;
            color: var(--text-light);
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: 1px;
          }
          
          .report-subtitle {
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
            color: var(--text-light);
            margin-bottom: 20px;
            opacity: 0.8;
            letter-spacing: 0.5px;
          }
          
          .report-meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            background: rgba(10, 31, 68, 0.5);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid rgba(0, 208, 255, 0.3);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
          }
          
          .meta-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: var(--text-light);
            font-family: 'Roboto', sans-serif;
          }
          
          .meta-icon {
            width: 16px;
            height: 16px;
            color: var(--accent-blue);
            filter: drop-shadow(0 0 3px rgba(0, 208, 255, 0.5));
          }
          
          /* Section Styles */
          .report-section {
            margin-bottom: 35px;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            border: 1px solid var(--glass-border);
            position: relative;
          }
          
          .section-header {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%);
            color: var(--text-light);
            padding: 18px 25px;
            font-family: 'Orbitron', sans-serif;
            font-size: 18px;
            font-weight: 600;
            letter-spacing: 1px;
            text-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
          }

          .section-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
              90deg,
              rgba(255, 255, 255, 0) 0%,
              rgba(255, 255, 255, 0.1) 50%,
              rgba(255, 255, 255, 0) 100%
            );
            animation: shimmer 3s infinite;
          }

          @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
          
          .section-content {
            padding: 30px 25px;
          }
          
          /* Results Grid */
          .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
          }
          
          .result-card {
            border: 1px solid rgba(0, 208, 255, 0.3);
            border-radius: 12px;
            padding: 25px;
            background: rgba(10, 31, 68, 0.3);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
          }

          .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%);
          }
          
          .result-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-light);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: 0.5px;
          }

          .result-title svg {
            color: var(--accent-blue);
            filter: drop-shadow(0 0 3px rgba(0, 208, 255, 0.5));
          }
          
          .result-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 22px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 20px;
            letter-spacing: 1px;
            text-shadow: 0 0 8px rgba(0, 208, 255, 0.4);
          }
          
          .confidence-container {
            margin-bottom: 10px;
          }
          
          .confidence-label {
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
            color: var(--text-light);
            margin-bottom: 8px;
            opacity: 0.8;
            letter-spacing: 0.3px;
          }
          
          .confidence-bar-container {
            background: rgba(0, 0, 0, 0.3);
            height: 14px;
            border-radius: 7px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(0, 208, 255, 0.2);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.3) inset;
          }
          
          .confidence-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%);
            border-radius: 7px;
            transition: width 0.3s ease;
            position: relative;
            box-shadow: 0 0 15px rgba(0, 208, 255, 0.6);
          }
          
          .confidence-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%);
            animation: confidence-shimmer 2s infinite;
          }
          
          @keyframes confidence-shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
          
          .confidence-percentage {
            font-family: 'Orbitron', sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-blue);
            margin-top: 6px;
            letter-spacing: 0.5px;
            text-shadow: 0 0 5px rgba(0, 208, 255, 0.3);
          }
          
          /* Explanation Section */
          .explanation-content {
            background: rgba(10, 31, 68, 0.4);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border-left: 4px solid var(--accent-blue);
            padding: 25px;
            border-radius: 0 12px 12px 0;
            font-family: 'Roboto', sans-serif;
            line-height: 1.8;
            color: var(--text-light);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            position: relative;
          }

          .explanation-content::before {
            content: '"';
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 60px;
            color: var(--accent-blue);
            opacity: 0.3;
            font-family: 'Orbitron', sans-serif;
            line-height: 1;
          }
          
          /* Image Section */
          .image-container {
            text-align: center;
            margin: 25px 0;
            position: relative;
          }
          
          .analysis-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            border: 2px solid rgba(0, 208, 255, 0.3);
            background: rgba(0, 0, 0, 0.2);
          }
          
          .image-caption {
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
            color: var(--text-light);
            margin-top: 12px;
            opacity: 0.7;
            letter-spacing: 0.3px;
          }
          
          /* Footer */
          .report-footer {
            margin-top: 50px;
            padding: 30px;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            position: relative;
          }

          .report-footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%);
            border-radius: 15px 15px 0 0;
          }
          
          .disclaimer {
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
          }
          
          .disclaimer-title {
            font-family: 'Orbitron', sans-serif;
            font-weight: 600;
            color: #ffc107;
            margin-bottom: 8px;
            font-size: 16px;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
          }
          
          .disclaimer-text {
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
            color: var(--text-light);
            line-height: 1.6;
            opacity: 0.9;
          }
          
          .copyright {
            text-align: center;
            font-family: 'Roboto', sans-serif;
            font-size: 12px;
            color: var(--text-light);
            opacity: 0.6;
            margin-top: 20px;
            letter-spacing: 0.5px;
          }

          .tech-pattern {
            position: absolute;
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            opacity: 0.1;
            font-family: 'Orbitron', monospace;
            font-size: 10px;
            color: var(--accent-blue);
            line-height: 1.2;
            pointer-events: none;
          }
          
          /* Print Optimizations */
          @media print {
            body {
              font-size: 12px;
              -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
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

            body::before {
              -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
            }

            .report-header,
            .report-section,
            .report-footer {
              -webkit-print-color-adjust: exact;
              print-color-adjust: exact;
            }
          }
          
          /* Responsive Design */
          @media (max-width: 768px) {
            .report-container {
              padding: 15px;
            }
            
            .section-content {
              padding: 20px 15px;
            }
            
            .results-grid {
              grid-template-columns: 1fr;
              gap: 20px;
            }
            
            .report-meta {
              grid-template-columns: 1fr;
              gap: 10px;
            }
            
            .logo {
              font-size: 24px;
            }
            
            .report-title {
              font-size: 20px;
            }
          }
        </style>
      </head>
      <body>
        <div class="report-container">
          <!-- Header -->
          <header class="report-header">
            <div class="logo">deepdiagnose</div>
            <h1 class="report-title">AI Medical Analysis Report</h1>
            <p class="report-subtitle">Advanced X-Ray Image Analysis System</p>
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
            <div class="tech-pattern">
              01001001 01000001<br/>
              11010001 01001000<br/>
              01000001 01001001<br/>
              10101010 11110000
            </div>
          </header>

          <!-- Analysis Results Section -->
          <section class="report-section">
            <div class="section-header">
              🔬 Analysis Results
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
              🧠 Expert Analysis
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
                <div class="image-caption">AI-generated overlay highlighting regions of interest in the medical image</div>
              </div>
            </div>
          </section>
          ` : ''}

          <!-- Footer -->
          <footer class="report-footer">
            <div class="disclaimer">
              <div class="disclaimer-title">
                ⚠️ Important Medical Disclaimer
              </div>
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
    a.download = `deepdiagnose-report-${new Date().toISOString().split('T')[0]}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
};