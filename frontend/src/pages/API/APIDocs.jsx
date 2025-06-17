import React, { useState } from 'react';
import Footer from '../../components/home-page/footer/footer';
import Header from '../../components/APIDocs/header/api-header'
import './APIDocs.css';
import { Copy, Check,  AlertCircle, CheckCircle, XCircle, } from 'lucide-react';

const APIDocs = () => {
  const [copiedEndpoint, setCopiedEndpoint] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(id);
    setTimeout(() => setCopiedEndpoint(''), 2000);
  };

  const CodeBlock = ({ children, language = 'javascript', copyId }) => (
    <div className="code-block">
      <div className="code-header">
        <span className="language-tag">{language}</span>
        <button 
          className="copy-button"
          onClick={() => copyToClipboard(children, copyId)}
        >
          {copiedEndpoint === copyId ? <Check size={16} /> : <Copy size={16} />}
        </button>
      </div>
      <pre><code>{children}</code></pre>
    </div>
  );

  const StatusBadge = ({ status, text }) => (
    <span className={`status-badge status-${status}`}>
      {status === 'success' && <CheckCircle size={14} />}
      {status === 'error' && <XCircle size={14} />}
      {status === 'warning' && <AlertCircle size={14} />}
      {text}
    </span>
  );

  return (
    <div className="api-docs-container">
        <Header />
      

      {/* Navigation Tabs */}
      <nav className="api-nav">
        <div className="nav-container">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'authentication', label: 'Authentication' },
            { id: 'endpoints', label: 'Endpoints' },
            { id: 'examples', label: 'Examples' },
            { id: 'errors', label: 'Error Handling' }
          ].map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content Sections */}
      <main className="api-content">
        {activeTab === 'overview' && (
          <section className="content-section">
            <h2>API Overview</h2>
            <p>The DeepDiagnose API provides advanced AI-powered medical image analysis capabilities. Our system can analyze X-rays and CT scans to detect various conditions including pneumonia, COVID-19, and osteoarthritis.</p>
            
            <div className="feature-grid">
              <div className="feature-card">
                <h3>Multi-Modal Analysis</h3>
                <p>Supports both X-ray and CT scan analysis with automatic scan type detection</p>
              </div>
              <div className="feature-card">
                <h3>AI-Powered Diagnostics</h3>
                <p>Advanced deep learning models trained on extensive medical imaging datasets</p>
              </div>
              <div className="feature-card">
                <h3>Visual Explanations</h3>
                <p>Grad-CAM heatmaps and overlays to visualize areas of interest</p>
              </div>
              <div className="feature-card">
                <h3>Patient-Friendly Reports</h3>
                <p>Automated generation of clear, understandable explanations</p>
              </div>
            </div>

            <div className="supported-conditions">
              <h3>Supported Conditions</h3>
              <div className="conditions-grid">
                <div className="condition">
                  <strong>Pneumonia Detection</strong>
                  <span>Chest X-ray analysis</span>
                </div>
                <div className="condition">
                  <strong>COVID-19 Detection</strong>
                  <span>CT scan analysis</span>
                </div>
                <div className="condition">
                  <strong>Osteoarthritis Detection</strong>
                  <span>Joint X-ray analysis</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'authentication' && (
          <section className="content-section">
            <h2>Authentication</h2>
            <p>Currently, the DeepDiagnose API operates without authentication for development purposes. In production, you would need to include your API key in the request headers.</p>
            
            <h3>Future Authentication (Production)</h3>
            <CodeBlock language="http" copyId="auth-header">
{`Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data`}
            </CodeBlock>

            <div className="auth-note">
              <AlertCircle className="note-icon" />
              <div>
                <strong>Development Note:</strong>
                <p>The current version doesn't require authentication. Contact our team to get your production API key when ready to deploy.</p>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'endpoints' && (
          <section className="content-section">
            <h2>API Endpoints</h2>
            
            <div className="endpoint-card">
              <div className="endpoint-header">
                <span className="method post">POST</span>
                <span className="endpoint-url">/predict</span>
              </div>
              <p>Upload and analyze medical images (X-rays or CT scans)</p>
              
              <h4>Parameters</h4>
              <div className="param-table">
                <div className="param-row header">
                  <span>Parameter</span>
                  <span>Type</span>
                  <span>Required</span>
                  <span>Description</span>
                </div>
                <div className="param-row">
                  <span><code>file</code></span>
                  <span>File</span>
                  <span className="required">Yes</span>
                  <span>Medical image file (JPEG, PNG, DICOM)</span>
                </div>
              </div>

              <h4>Image Requirements</h4>
              <ul className="requirements-list">
                <li>Minimum dimensions: 64x64 pixels</li>
                <li>Maximum dimensions: 8000x8000 pixels</li>
                <li>Supported formats: JPEG, PNG, DICOM</li>
                <li>Must be grayscale or near-grayscale medical images</li>
                <li>X-ray or CT scan images only</li>
                <li>Maximum file size: 50MB</li>
              </ul>

              <h4>Response Format</h4>
              <CodeBlock language="json" copyId="response-format">
{`{
  "scan_type": "X-ray",
  "scan_type_confidence": 0.956,
  "anatomy": "Chest-scan",
  "anatomy_confidence": 0.923,
  "disease": "Pneumonia",
  "disease_confidence": 0.847,
  "overlay_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "explanation": "The analysis indicates signs consistent with pneumonia..."
}`}
              </CodeBlock>

              <h4>Response Fields</h4>
              <div className="response-fields">
                <div className="field">
                  <code>scan_type</code>
                  <span>Detected medical scan type ("X-ray" or "CT")</span>
                </div>
                <div className="field">
                  <code>scan_type_confidence</code>
                  <span>Confidence score for scan type classification (0-1)</span>
                </div>
                <div className="field">
                  <code>anatomy</code>
                  <span>Anatomical region ("Chest-scan", "Joint-scan", or "CT Scan")</span>
                </div>
                <div className="field">
                  <code>anatomy_confidence</code>
                  <span>Confidence score for anatomical classification (0-1)</span>
                </div>
                <div className="field">
                  <code>disease</code>
                  <span>Detected condition ("Normal", "Pneumonia", "COVID-19", "Osteoarthritis")</span>
                </div>
                <div className="field">
                  <code>disease_confidence</code>
                  <span>Confidence score for disease classification (0-1)</span>
                </div>
                <div className="field">
                  <code>overlay_image</code>
                  <span>Base64-encoded Grad-CAM visualization overlay</span>
                </div>
                <div className="field">
                  <code>explanation</code>
                  <span>Patient-friendly explanation of the analysis results</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'examples' && (
          <section className="content-section">
            <h2>Code Examples</h2>
            
            <h3>JavaScript (Fetch API)</h3>
            <CodeBlock language="javascript" copyId="js-example">
{`const analyzeImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(\`HTTP error! status: \${response.status}\`);
    }

    const result = await response.json();
    console.log('Analysis result:', result);
    
    // Display the overlay image
    const imgElement = document.getElementById('overlay');
    imgElement.src = result.overlay_image;
    
    return result;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

// Usage
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) {
    const result = await analyzeImage(file);
    // Handle the result
  }
});`}
            </CodeBlock>

            <h3>Python (requests)</h3>
            <CodeBlock language="python" copyId="python-example">
{`import requests
import base64
from PIL import Image
from io import BytesIO

def analyze_medical_image(image_path):
    """
    Analyze a medical image using the DeepDiagnose API
    """
    url = "http://localhost:8000/predict"
    
    try:
        with open(image_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # Save the overlay image if provided
            if result.get('overlay_image'):
                overlay_data = result['overlay_image'].split(',')[1]
                overlay_bytes = base64.b64decode(overlay_data)
                overlay_image = Image.open(BytesIO(overlay_bytes))
                overlay_image.save('analysis_overlay.png')
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

# Usage
result = analyze_medical_image('chest_xray.jpg')
if result:
    print(f"Scan Type: {result['scan_type']}")
    print(f"Anatomy: {result['anatomy']}")
    print(f"Diagnosis: {result['disease']}")
    print(f"Confidence: {result['disease_confidence']:.2%}")
    print(f"Explanation: {result['explanation']}")`}
            </CodeBlock>

            <h3>cURL</h3>
            <CodeBlock language="bash" copyId="curl-example">
{`curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@/path/to/your/medical_image.jpg"`}
            </CodeBlock>

            <h3>React Component Example</h3>
            <CodeBlock language="jsx" copyId="react-example">
{`import React, { useState } from 'react';

const MedicalImageAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyzer">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />
      <button onClick={analyzeImage} disabled={!file || loading}>
        {loading ? 'Analyzing...' : 'Analyze Image'}
      </button>

      {error && (
        <div className="error">
          <p>Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="results">
          <h3>Analysis Results</h3>
          <p><strong>Scan Type:</strong> {result.scan_type}</p>
          <p><strong>Anatomy:</strong> {result.anatomy}</p>
          <p><strong>Diagnosis:</strong> {result.disease}</p>
          <p><strong>Confidence:</strong> {(result.disease_confidence * 100).toFixed(1)}%</p>
          
          {result.overlay_image && (
            <div className="overlay">
              <h4>Analysis Visualization</h4>
              <img src={result.overlay_image} alt="Analysis overlay" />
            </div>
          )}
          
          <div className="explanation">
            <h4>Explanation</h4>
            <p>{result.explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default MedicalImageAnalyzer;`}
            </CodeBlock>
          </section>
        )}

        {activeTab === 'errors' && (
          <section className="content-section">
            <h2>Error Handling</h2>
            <p>The API uses standard HTTP status codes to indicate the success or failure of requests. Here are the common error responses:</p>

            <div className="error-codes">
              <div className="error-code">
                <div className="error-header">
                  <StatusBadge status="error" text="400 Bad Request" />
                </div>
                <p>Invalid image file or unsupported medical image type</p>
                <CodeBlock language="json" copyId="error-400">
{`{
  "error": "Invalid medical image: Image appears to be a colored photo, not a medical scan. Please upload a valid X-ray or CT scan."
}`}
                </CodeBlock>
              </div>

              <div className="error-code">
                <div className="error-header">
                  <StatusBadge status="error" text="422 Unprocessable Entity" />
                </div>
                <p>Image validation failed or processing requirements not met</p>
                <CodeBlock language="json" copyId="error-422">
{`{
  "error": "Unable to determine scan type with sufficient confidence. Please upload a clearer medical image."
}`}
                </CodeBlock>
              </div>

              <div className="error-code">
                <div className="error-header">
                  <StatusBadge status="error" text="500 Internal Server Error" />
                </div>
                <p>Server error during processing</p>
                <CodeBlock language="json" copyId="error-500">
{`{
  "error": "Unexpected error during prediction: Model processing failed"
}`}
                </CodeBlock>
              </div>
            </div>

            <h3>Common Error Scenarios</h3>
            <div className="error-scenarios">
              <div className="scenario">
                <h4>Invalid Medical Image</h4>
                <p>The API performs rigorous validation to ensure uploaded images are medical scans. Common rejection reasons:</p>
                <ul>
                  <li>Colored photographs instead of medical scans</li>
                  <li>Images with high color saturation</li>
                  <li>Non-medical images (documents, drawings, etc.)</li>
                  <li>Images lacking anatomical structure</li>
                </ul>
              </div>

              <div className="scenario">
                <h4>Low Confidence Predictions</h4>
                <p>When the AI model's confidence is below the threshold (80%), the API returns an error encouraging professional consultation:</p>
                <ul>
                  <li>Scan type classification confidence &lt; 80%</li>
                  <li>Anatomy classification confidence &lt; 80%</li>
                  <li>Disease classification confidence &lt; 80%</li>
                  <li>COVID-19 predictions requiring 90% confidence</li>
                </ul>
              </div>

              <div className="scenario">
                <h4>File Format Issues</h4>
                <p>Ensure your images meet the technical requirements:</p>
                <ul>
                  <li>Supported formats: JPEG, PNG, DICOM</li>
                  <li>Minimum size: 64x64 pixels</li>
                  <li>Maximum size: 8000x8000 pixels</li>
                  <li>File size limit: 50MB</li>
                </ul>
              </div>
            </div>

            <h3>Best Practices</h3>
            <div className="best-practices">
              <div className="practice">
                <CheckCircle className="practice-icon" />
                <div>
                  <strong>Always handle errors gracefully</strong>
                  <p>Implement proper error handling in your application to provide meaningful feedback to users.</p>
                </div>
              </div>
              <div className="practice">
                <CheckCircle className="practice-icon" />
                <div>
                  <strong>Validate images client-side</strong>
                  <p>Pre-validate image format and size before sending to reduce unnecessary API calls.</p>
                </div>
              </div>
              <div className="practice">
                <CheckCircle className="practice-icon" />
                <div>
                  <strong>Provide user guidance</strong>
                  <p>When errors occur, guide users on how to obtain suitable medical images.</p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
        <Footer />
    </div>
        
  );
    
    
};

export default APIDocs;