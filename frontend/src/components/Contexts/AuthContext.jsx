import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  // API base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Check if token is expired
  const isTokenExpired = (token) => {
    if (!token) return true;
    
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      return payload.exp < currentTime;
    } catch (error) {
      return true;
    }
  };

  // Clear all auth storage
  const clearAuthStorage = useCallback(() => {
    // Clear all possible token/user storage keys
    ['authToken', 'access_token', 'userData', 'user'].forEach(key => {
      localStorage.removeItem(key);
      sessionStorage.removeItem(key);
    });
  }, []);

  // Verify token with backend
  const verifyToken = useCallback(async (token) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const userData = await response.json();
        return userData;
      }
      return null;
    } catch (error) {
      console.error('Token verification failed:', error);
      return null;
    }
  }, [API_BASE_URL]);

  // Initialize auth state on app load
  useEffect(() => {
    const initializeAuth = async () => {
      console.log('🔍 Initializing auth...');
      
      // Check both localStorage and sessionStorage for token
      const storedToken = localStorage.getItem('authToken') || 
                         localStorage.getItem('access_token') ||
                         sessionStorage.getItem('authToken') ||
                         sessionStorage.getItem('access_token');
      
      const storedUser = localStorage.getItem('userData') || 
                        localStorage.getItem('user') ||
                        sessionStorage.getItem('userData') ||
                        sessionStorage.getItem('user');

      console.log('🔑 Found stored token:', !!storedToken);
      console.log('👤 Found stored user:', !!storedUser);

      if (storedToken && !isTokenExpired(storedToken)) {
        // Try cached user data first
        if (storedUser) {
          try {
            const cachedUser = JSON.parse(storedUser);
            setUser(cachedUser);
            setToken(storedToken);
          } catch (e) {
            console.error('Failed to parse cached user data:', e);
          }
        }

        // Verify token with backend
        const userData = await verifyToken(storedToken);
        
        if (userData) {
          setToken(storedToken);
          setUser(userData);
          
          // Update stored user data
          const storage = localStorage.getItem('authToken') || localStorage.getItem('access_token') 
                         ? localStorage : sessionStorage;
          storage.setItem('userData', JSON.stringify(userData));
          
          // Auto-redirect to /upload if on home page and authenticated
          if (location.pathname === '/' || location.pathname === '/login') {
            console.log('🔄 Auto-redirecting to /upload');
            navigate('/upload', { replace: true });
          }
        } else {
          // Token is invalid, clear all storage
          console.log('❌ Token invalid, clearing storage');
          clearAuthStorage();
        }
      } else if (storedToken) {
        // Token exists but is expired
        console.log('⏰ Token expired, clearing storage');
        clearAuthStorage();
      }

      setLoading(false);
    };

    initializeAuth();
  }, [navigate, location.pathname, verifyToken, clearAuthStorage]);

  const login = async (credentials) => {
    try {
      console.log('🔐 Attempting login...');
      
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: credentials.email || credentials.username, // Backend expects 'username'
          password: credentials.password
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Login successful:', data);
        
        // Store token and user data (use primary keys)
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('authToken', data.access_token); // Keep compatibility
        
        if (data.user) {
          localStorage.setItem('user', JSON.stringify(data.user));
          localStorage.setItem('userData', JSON.stringify(data.user)); // Keep compatibility
        }
        
        setToken(data.access_token);
        setUser(data.user);

        // Redirect to upload page
        navigate('/upload', { replace: true });
        
        return { success: true, data };
      } else {
        const errorData = await response.json();
        console.log('❌ Login failed:', errorData);
        
        let errorMessage = 'Login failed';
        if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map(err => err.msg).join('. ');
        } else if (errorData.detail) {
          errorMessage = errorData.detail;
        }
        
        return { success: false, error: errorMessage };
      }
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const signup = async (userData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        const errorData = await response.json();
        return { success: false, error: errorData.detail || 'Signup failed' };
      }
    } catch (error) {
      console.error('Signup error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const logout = () => {
    console.log('🚪 Logging out...');
    
    // Clear all auth data
    clearAuthStorage();
    setToken(null);
    setUser(null);
    
    // Redirect to home page
    navigate('/', { replace: true });
    
    console.log('✅ Logout complete');
  };

  const forgotPassword = async (email) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error('Forgot password error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const verifyResetCode = async (email, code) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/verify-reset-code`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, code }),
      });

      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error('Verify code error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const resetPassword = async (email, code, newPassword) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          email, 
          code, 
          new_password: newPassword 
        }),
      });

      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error('Reset password error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const value = {
    user,
    token,
    loading,
    isAuthenticated: !!token && !!user,
    login,
    signup,
    logout,
    forgotPassword,
    verifyResetCode,
    resetPassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};