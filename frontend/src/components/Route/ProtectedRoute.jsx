import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../Contexts/AuthContext'; 
import './ProtectedRoute.css'; 

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking authentication
if (loading) {
  return (
    <div className="protected-loading">
      <div className="protected-loading-content">
        <div className="loading-spinner"></div>
        <p className="loading-text">Authenticating...</p>
      </div>
    </div>
  );
}

  // If not authenticated, redirect to home with the attempted location
  if (!isAuthenticated) {
    return <Navigate to="/" state={{ from: location }} replace />;
  }

  // If authenticated, render the protected component
  return children;
};

export default ProtectedRoute;