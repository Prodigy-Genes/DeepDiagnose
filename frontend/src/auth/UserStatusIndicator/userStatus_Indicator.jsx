import React, { useState, useEffect } from 'react';
import './userStatus_Indicator.css';

const UserStatusIndicator = ({ onLoginClick }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    // Check if user is authenticated on component mount
    checkAuthStatus();
    
    // Set up online/offline detection
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      
      if (!token) {
        setIsLoading(false);
        return;
      }

      // In a real app, you'd verify the token with your backend
      // For now, we'll check if token exists and try to decode user info
      const response = await fetch('http://localhost:8000/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        // Token is invalid, remove it
        localStorage.removeItem('authToken');
        sessionStorage.removeItem('authToken');
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      // If API is unavailable, try to get user data from token or localStorage
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      const userData = localStorage.getItem('userData');
      
      if (token && userData) {
        try {
          setUser(JSON.parse(userData));
        } catch (e) {
          console.error('Failed to parse user data:', e);
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    sessionStorage.removeItem('authToken');
    setUser(null);
    setShowDropdown(false);
    
    // Optionally redirect to login or refresh page
    window.location.reload();
  };

  const handleLogin = () => {
    if (onLoginClick) {
      onLoginClick();
    } else {
      // Fallback: redirect to login page or show login modal
      console.log('Login clicked - implement your login logic here');
    }
  };

  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
  };

  // Click outside handler
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showDropdown && !event.target.closest('.user-status-indicator')) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdown]);

  if (isLoading) {
    return (
      <div className="user-status-indicator loading">
        <div className="status-loader">
          <i className="fas fa-spinner fa-spin"></i>
        </div>
      </div>
    );
  }

  // Show login button when user is not authenticated
  if (!user) {
    return (
      <div className="user-status-indicator">
        <button className="login-button" onClick={handleLogin}>
          <i className="fas fa-sign-in-alt"></i>
          <span>Sign In</span>
        </button>
      </div>
    );
  }

  return (
    <div className="user-status-indicator">
      <div className="user-info-container" onClick={toggleDropdown}>
        <div className="user-avatar">
          <i className="fas fa-user-md"></i>
          <div className={`online-status ${isOnline ? 'online' : 'offline'}`}></div>
        </div>
        
        <div className="user-details">
          <div className="user-name">{user.username || user.name || 'User'}</div>
          <div className="user-status">
            <i className="fas fa-circle status-dot"></i>
            <span>Authenticated</span>
          </div>
        </div>
        
        <div className="dropdown-arrow">
          <i className={`fas fa-chevron-${showDropdown ? 'up' : 'down'}`}></i>
        </div>
      </div>

      {/* Dropdown Menu */}
      {showDropdown && (
        <div className="user-dropdown">
          <div className="dropdown-header">
            <div className="user-email">{user.email}</div>
            <div className="user-role">
              <i className="fas fa-stethoscope"></i>
              Medical Professional
            </div>
          </div>
          
          <div className="dropdown-divider"></div>
          
          <div className="dropdown-items">
            <button className="dropdown-item">
              <i className="fas fa-user-cog"></i>
              Profile Settings
            </button>
            <button className="dropdown-item">
              <i className="fas fa-history"></i>
              Analysis History
            </button>
            <button className="dropdown-item">
              <i className="fas fa-shield-alt"></i>
              Privacy & Security
            </button>
          </div>
          
          <div className="dropdown-divider"></div>
          
          <div className="dropdown-footer">
            <button className="logout-btn" onClick={handleLogout}>
              <i className="fas fa-sign-out-alt"></i>
              Sign Out
            </button>
          </div>
        </div>
      )}

      {/* Connection Status Indicator */}
      {!isOnline && (
        <div className="connection-warning">
          <i className="fas fa-wifi-slash"></i>
          <span>Offline Mode</span>
        </div>
      )}
    </div>
  );
};

export default UserStatusIndicator;