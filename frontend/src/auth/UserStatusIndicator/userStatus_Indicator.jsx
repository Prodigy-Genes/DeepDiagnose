import React, { useState, useEffect } from 'react';
import './userStatus_Indicator.css';
import MedicalHistoryList from '../../components/MedicalHistoryList/MedicalHistoryList';
import MedicalStatistics from '../../components/MedicalStatistics/MedicalStatistics';

const UserStatusIndicator = ({ onLoginClick }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [token, setToken] = useState(null); // Add token state

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
    console.log('🔍 Checking auth status...');
    
    try {
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      setToken(token);
      const storedUserData = localStorage.getItem('userData') || sessionStorage.getItem('userData');
      
      console.log('🔑 Token exists:', !!token);
      console.log('👤 Stored user data exists:', !!storedUserData);
      
      if (!token) {
        console.log('❌ No token found, user not authenticated');
        setIsLoading(false);
        return;
      }

      // Try to use cached user data first for faster loading
      if (storedUserData) {
        try {
          const cachedUser = JSON.parse(storedUserData);
          console.log('⚡ Using cached user data:', cachedUser);
          setUser(cachedUser);
        } catch (e) {
          console.error('❌ Failed to parse cached user data:', e);
        }
      }

      // Verify token with backend
      console.log('🔄 Verifying token with backend...');
      const response = await fetch('http://localhost:8000/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      console.log('📝 Backend response status:', response.status);

      if (response.ok) {
        const userData = await response.json();
        console.log('✅ Backend verification successful:', userData);
        setUser(userData);
        
        // Update stored user data if it's different
        const storage = localStorage.getItem('authToken') ? localStorage : sessionStorage;
        storage.setItem('userData', JSON.stringify(userData));
      } else {
        console.log('❌ Backend verification failed, clearing tokens');
        // Token is invalid, remove it
        localStorage.removeItem('authToken');
        localStorage.removeItem('userData');
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('userData');
        setUser(null);
      }
    } catch (error) {
      console.error('🚨 Auth check failed with error:', error);
      
      // If API is unavailable, try to use cached user data
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      const userData = localStorage.getItem('userData') || sessionStorage.getItem('userData');
      
      if (token && userData) {
        try {
          console.log('🔄 Using cached data as fallback due to network error');
          setUser(JSON.parse(userData));
        } catch (e) {
          console.error('❌ Failed to parse cached user data:', e);
        }
      }
    } finally {
      console.log('✅ Auth check complete');
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    console.log('🚪 Logging out user...');
    
    // Clear all authentication data
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    sessionStorage.removeItem('authToken');
    sessionStorage.removeItem('userData');
    
    // Reset component state
    setUser(null);
    setShowDropdown(false);
    
    console.log('✅ Logout complete, reloading page...');
    
    // Force a complete page reload to reset all component states
    window.location.reload();
  };

  const handleLogin = () => {
    console.log('🔐 Login button clicked');
    console.log('🔗 onLoginClick callback exists:', !!onLoginClick);
    
    if (onLoginClick) {
      console.log('📞 Calling onLoginClick callback');
      try {
        onLoginClick();
      } catch (error) {
        console.error('❌ Error calling onLoginClick:', error);
        // Fallback to manual navigation
        handleFallbackLogin();
      }
    } else {
      console.log('⚠️ No onLoginClick callback provided, using fallback');
      handleFallbackLogin();
    }
  };

  const handleFallbackLogin = () => {
    console.log('🔄 Using fallback login method');
    
    // Try to dispatch a custom event that parent components can listen for
    const loginEvent = new CustomEvent('requestLogin', {
      bubbles: true,
      detail: { source: 'UserStatusIndicator' }
    });
    
    document.dispatchEvent(loginEvent);
    
    // If no event handlers, show an alert or redirect
    setTimeout(() => {
      const isStillLoggedOut = !localStorage.getItem('authToken') && !sessionStorage.getItem('authToken');
      if (isStillLoggedOut) {
        console.log('🚨 No login handler responded, showing alert');
        alert('Please refresh the page and try logging in again.');
      }
    }, 1000);
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

  // Add listener for storage changes (useful for multi-tab scenarios)
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === 'authToken' || e.key === 'userData') {
        console.log('🔄 Storage change detected, rechecking auth status');
        checkAuthStatus();
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

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
    console.log('🔓 Rendering login button - user not authenticated');
    return (
      <div className="user-status-indicator">
        <button 
          className="login-button" 
          onClick={handleLogin}
          style={{ cursor: 'pointer' }} // Ensure it's clickable
        >
          <i className="fas fa-sign-in-alt"></i>
          <span>Sign In</span>
        </button>
      </div>
    );
  }

  console.log('🔒 Rendering authenticated user interface for:', user.username || user.email);

  return (
    <div className="user-status-indicator">
      <div className="user-info-container" onClick={toggleDropdown}>
        <div className="user-avatar">
          <i className="fas fa-user-md"></i>
          <div className={`online-status ${isOnline ? 'online' : 'offline'}`}></div>
        </div>
        
        <div className="user-details">
          <div className="user-name">{user.username || user.name || user.email?.split('@')[0] || 'User'}</div>
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
          
          <MedicalStatistics token={token} />
          <MedicalHistoryList token={token} />
          
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