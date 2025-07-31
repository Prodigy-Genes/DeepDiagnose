import React, { useState, useEffect } from 'react';
import './userStatus_Indicator.css';
import MedicalStatisticsModal from '../../components/MedicalStatisticsModal/MedicalStatisticsModal';
import MedicalHistoryModal from '../../components/MedicalHistoryListModal/MedicalHistoryListModal';
import { useAuth } from '../../components/Contexts/AuthContext';

const UserStatusIndicator = ({ onLoginClick }) => {
  const [showDropdown, setShowDropdown] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  
  // Modal states
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);

  // Get auth state from context
  const { user, token, isAuthenticated, loading, logout } = useAuth();

  useEffect(() => {
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

  const handleLogout = () => {
    console.log('🚪 Logging out user...');
    
    // Use AuthContext logout - this handles all cleanup and routing
    logout();
    
    // Close dropdown
    setShowDropdown(false);
    
    console.log('✅ Logout complete');
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
      if (!isAuthenticated) {
        console.log('🚨 No login handler responded, showing alert');
        alert('Please refresh the page and try logging in again.');
      }
    }, 1000);
  };

  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
  };

  const handleMenuItemClick = (action) => {
    setShowDropdown(false); // Close dropdown first
    
    switch (action) {
      case 'statistics':
        setShowStatsModal(true);
        break;
      case 'history':
        setShowHistoryModal(true);
        break;
      case 'profile':
        // Handle profile action
        console.log('Profile clicked');
        break;
      case 'settings':
        // Handle settings action
        console.log('Settings clicked');
        break;
      default:
        break;
    }
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

  // Show loading state
  if (loading) {
    return (
      <div className="user-status-indicator loading">
        <div className="status-loader">
          <i className="fas fa-spinner fa-spin"></i>
        </div>
      </div>
    );
  }

  // Show login button when user is not authenticated
  if (!isAuthenticated || !user) {
    console.log('🔓 Rendering login button - user not authenticated');
    return (
      <div className="user-status-indicator">
        <button 
          className="login-button" 
          onClick={handleLogin}
          style={{ cursor: 'pointer' }}
        >
          <i className="fas fa-sign-in-alt"></i>
          <span>Sign In</span>
        </button>
      </div>
    );
  }

  console.log('🔒 Rendering authenticated user interface for:', user.username || user.email);

  return (
    <>
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

        {/* Compact Dropdown Menu */}
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
            
            <div className="dropdown-menu">
              <button 
                className="dropdown-menu-item"
                onClick={() => handleMenuItemClick('statistics')}
              >
                <i className="fas fa-chart-bar"></i>
                <span>View Statistics</span>
                <i className="fas fa-external-link-alt"></i>
              </button>
              
              <button 
                className="dropdown-menu-item"
                onClick={() => handleMenuItemClick('history')}
              >
                <i className="fas fa-history"></i>
                <span>Medical Scan History</span>
                <i className="fas fa-external-link-alt"></i>
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

      {/* Modals */}
      <MedicalStatisticsModal 
        isOpen={showStatsModal}
        onClose={() => setShowStatsModal(false)}
        token={token}
      />
      
      <MedicalHistoryModal
        isOpen={showHistoryModal}
        onClose={() => setShowHistoryModal(false)}
        token={token}
      />
    </>
  );
};

export default UserStatusIndicator;