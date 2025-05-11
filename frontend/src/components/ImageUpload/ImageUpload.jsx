import React, { useState, useRef } from "react";
import axios from "axios";
import "./ImageUpload.css";

// This component allows users to upload an image and get a prediction from the server
const ImageUpload = ({ onResult }) => {
  const [image, setImage] = useState(null); // State to manage the uploaded image
  const [loading, setLoading] = useState(false); // State to manage loading state
  const [dragActive, setDragActive] = useState(false); // State to manage drag and drop
  const [previewUrl, setPreviewUrl] = useState(null); // State to manage the preview URL of the image
  const fileInputRef = useRef(null); // Ref to manage the file input element

  const handleFileChange = (file) => {
    if (!file) return;
    
    setImage(file);
    
    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
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
    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      onResult(response.data);
    } catch (err) {
      alert("Prediction failed! Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const resetImage = () => {
    setImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="image-upload-container">
      <div className="upload-card">
        <h2 className="upload-title">Image Analysis</h2>
        
        {!previewUrl ? (
          <div 
            className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
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
              accept="image/*"
              className="file-input"
            />
            <div className="dropzone-content">
              <svg 
                className="upload-icon" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24" 
                xmlns="http://www.w3.org/2000/svg"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p className="dropzone-text">Drag and drop an image here, or click to select</p>
              <p className="dropzone-hint">JPG, PNG, GIF up to 10MB</p>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            <div className="image-preview-wrapper">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="image-preview"
              />
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
            <p className="file-info">
              {image.name} ({(image.size / 1024).toFixed(1)} KB)
            </p>
          </div>
        )}
        
        <div className="button-container">
          <button
            onClick={handleSubmit}
            disabled={!image || loading}
            className={`analyze-button ${(!image || loading) ? "button-disabled" : ""}`}
          >
            {loading ? (
              <>
                <svg className="spinner" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="spinner-track" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="spinner-path" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              "Analyze Image"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;