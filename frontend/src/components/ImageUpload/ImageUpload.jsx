import React, { useState, useRef } from "react";
import "./ImageUpload.css";

// This component allows users to upload an image and get a prediction from the server
const ImageUpload = ({ onResult, onUploadStart }) => {
  const [image, setImage] = useState(null); // State to manage the uploaded image
  const [loading, setLoading] = useState(false); // State to manage loading state
  const [dragActive, setDragActive] = useState(false); // State to manage drag and drop
  const [previewUrl, setPreviewUrl] = useState(null); // State to manage the preview URL of the image
  const [error, setError] = useState(null); // State to manage error messages
  const fileInputRef = useRef(null); // Ref to manage the file input element

  const handleFileChange = (file) => {
    if (!file) return;
    
    // Clear any existing errors when a new file is selected
    setError(null);
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/dicom'];
    if (!validTypes.includes(file.type) && !file.name.endsWith('.dcm')) {
      setError("Please upload a valid image (JPEG, PNG or DICOM format)");
      return;
    }
    
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit");
      return;
    }
    
    setImage(file);
    
    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.onerror = () => {
      setError("Failed to read file. Please try again.");
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append("file", image);
    
    setLoading(true);
    setError(null);
    
    // Notify parent component that upload has started
    if (onUploadStart) {
      onUploadStart();
    }
    
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Server error. Please try again.");
      }
      
      const data = await response.json();
      onResult(data);
    } catch (err) {
      // Handle different types of errors
      if (err.message.includes("X-ray")) {
        setError("This doesn't appear to be an X-ray image. Please upload a valid X-ray scan.");
      } else if (err.message.includes("clearer image")) {
        setError("The image quality is too low for analysis. Please upload a clearer X-ray image.");
      } else if (err.name === "AbortError") {
        setError("Request timed out. Please check your connection and try again.");
      } else if (!navigator.onLine) {
        setError("You appear to be offline. Please check your internet connection.");
      } else {
        setError(err.message || "Analysis failed. Please try again with a different image.");
      }
    } finally {
      setLoading(false);
    }
  };

  const resetImage = () => {
    setImage(null);
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="image-upload-container">
      <div className="upload-card">
        {error && (
          <div className="error-message">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-11v4h2v-4h-2zm0-6v2h2V5h-2z"
                fill="currentColor"
              />
            </svg>
            <span>{error}</span>
            <button
              className="dismiss-error"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6 18L18 6M6 6l12 12"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        )}

        {!previewUrl ? (
          <div
            className={`dropzone ${dragActive ? "dropzone-active" : ""} ${error ? "dropzone-error" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={handleClick}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileChange(e.target.files[0])}
              accept="image/*,.dcm"
              className="file-input"
            />
            <div className="dropzone-content">
              <div className="lungs-icon">
                <svg
                  width="60"
                  height="60"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12 2C10.5 2 9.5 3 9.5 4.5V9.5C9.5 11 6 13 6 17C6 19.5 7.5 22 11 22C13 22 14 21 14 19.5V13"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M12 2C13.5 2 14.5 3 14.5 4.5V9.5C14.5 11 18 13 18 17C18 19.5 16.5 22 13 22C11 22 10 21 10 19.5V13"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <div className="dropzone-text-container">
                <p className="dropzone-title">Upload X-Ray Image</p>
                <p className="dropzone-text">Drag and drop an x-ray image here, or click to select</p>
                <p className="dropzone-hint">DICOM, JPG, PNG formats supported up to 10MB</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            <div className={`image-preview-wrapper ${error ? "preview-error" : ""}`}>
              <img
                src={previewUrl}
                alt="Preview"
                className="image-preview"
              />
              <div className="overlay-grid">
                {[...Array(9)].map((_, index) => (
                  <div key={index} className="grid-square"></div>
                ))}
              </div>
              <div className="image-stats">
                <div className="stat-item">
                  <span className="stat-label">Resolution</span>
                  <span className="stat-value">Standard</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Type</span>
                  <span className="stat-value">X-Ray</span>
                </div>
              </div>
              <button
                onClick={resetImage}
                className="remove-button"
                title="Remove image"
              >
                <svg
                  className="remove-icon"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <div className="file-info-container">
              <div className="file-info">
                <i className="fas fa-file-medical"></i>
                <span>
                  {image.name} ({(image.size / 1024).toFixed(1)} KB)
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="button-container"></div>
        <button
          onClick={handleSubmit}
          disabled={!image || loading}
          className={`analyze-button ${(!image || loading) ? "button-disabled" : ""}`}
        >
          {!loading ? (
            <>
              <span className="button-icon">
                <i className="fas fa-brain"></i>
              </span>
              <span>Analyze with AI</span>
            </>
          ) : (
            <span>Initializing...</span>
          )}
        </button>

        {/* Processing visualization - shown when loading */}
        {loading && (
          <div className="processing-visualization">
            <div className="processing-grid">
              {[...Array(25)].map((_, index) => (
                <div key={index} className="grid-cell"></div>
              ))}
            </div>
            <div className="processing-text">
              <div className="binary-code">
                01001100 01001111 01000001 01000100 01001001 01001110 01000111
              </div>
              <div className="analyzing-text">Analyzing X-ray patterns...</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;