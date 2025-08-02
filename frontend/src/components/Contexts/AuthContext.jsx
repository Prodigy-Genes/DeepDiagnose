import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
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
  const [initialized, setInitialized] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  
  // Use ref to prevent infinite re-initialization
  const initializationRef = useRef(false);
  const tokenVerificationRef = useRef(null);

  // API base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Check if token is expired
  const isTokenExpired = useCallback((token) => {
    if (!token) return true;
    
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      // Add 5 minute buffer to prevent edge cases
      return payload.exp < (currentTime + 300);
    } catch (error) {
      console.error('Token parsing error:', error);
      return true;
    }
  }, []);

  // Clear all auth storage
  const clearAuthStorage = useCallback(() => {
    console.log('🧹 Clearing auth storage...');
    
    // Clear all possible token/user storage keys
    const keys = ['authToken', 'access_token', 'userData', 'user'];
    keys.forEach(key => {
      localStorage.removeItem(key);
      sessionStorage.removeItem(key);
    });
    
    // Clear verification cache
    tokenVerificationRef.current = null;
  }, []);

  // Verify token with backend (with caching to prevent repeated calls)
  const verifyToken = useCallback(async (token) => {
    // Prevent duplicate verification calls
    if (tokenVerificationRef.current === token) {
      console.log('🔄 Token verification already in progress');
      return null;
    }
    
    tokenVerificationRef.current = token;
    
    try {
      console.log('🔍 Verifying token with backend...');
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
      
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const userData = await response.json();
        console.log('✅ Token verified successfully');
        return userData;
      } else {
        console.log('❌ Token verification failed:', response.status);
        return null;
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('🕐 Token verification timeout');
      } else {
        console.error('❌ Token verification error:', error);
      }
      return null;
    } finally {
      tokenVerificationRef.current = null;
    }
  }, [API_BASE_URL]);

  // Get stored auth data
  const getStoredAuthData = useCallback(() => {
    const storedToken = localStorage.getItem('access_token') || 
                       localStorage.getItem('authToken') ||
                       sessionStorage.getItem('access_token') ||
                       sessionStorage.getItem('authToken');
    
    const storedUser = localStorage.getItem('user') || 
                      localStorage.getItem('userData') ||
                      sessionStorage.getItem('user') ||
                      sessionStorage.getItem('userData');

    return { storedToken, storedUser };
  }, []);

  // Initialize auth state on app load
  useEffect(() => {
    // Prevent multiple initializations
    if (initializationRef.current) {
      console.log('⚠️ Auth already initialized, skipping...');
      return;
    }

    const initializeAuth = async () => {
      console.log('🔍 Initializing auth...');
      initializationRef.current = true;
      
      try {
        const { storedToken, storedUser } = getStoredAuthData();
        
        console.log('🔑 Found stored token:', !!storedToken);
        console.log('👤 Found stored user:', !!storedUser);

        if (!storedToken) {
          console.log('📭 No stored token found');
          setLoading(false);
          setInitialized(true);
          return;
        }

        if (isTokenExpired(storedToken)) {
          console.log('⏰ Token expired, clearing storage');
          clearAuthStorage();
          setLoading(false);
          setInitialized(true);
          return;
        }

        // Set cached data first for immediate UI update
        if (storedUser) {
          try {
            const cachedUser = JSON.parse(storedUser);
            setUser(cachedUser);
            setToken(storedToken);
            console.log('📦 Using cached user data');
          } catch (e) {
            console.error('❌ Failed to parse cached user data:', e);
          }
        }

        // Verify token with backend
        const userData = await verifyToken(storedToken);
        
        if (userData) {
          console.log('✅ Token verification successful');
          setToken(storedToken);
          setUser(userData);
          
          // Update stored user data if different
          const currentUserString = JSON.stringify(userData);
          if (storedUser !== currentUserString) {
            const storage = localStorage.getItem('access_token') ? localStorage : sessionStorage;
            storage.setItem('user', currentUserString);
          }
          
        } else {
          console.log('❌ Token invalid, clearing auth state');
          clearAuthStorage();
          setToken(null);
          setUser(null);
        }
      } catch (error) {
        console.error('❌ Auth initialization error:', error);
        clearAuthStorage();
        setToken(null);
        setUser(null);
      } finally {
        setLoading(false);
        setInitialized(true);
      }
    };

    initializeAuth();
  }, ); // Remove dependencies to prevent re-initialization

  // Handle navigation after auth state is established
  useEffect(() => {
    if (!initialized || loading) return;

    const isAuthenticated = !!token && !!user && !isTokenExpired(token);
    const isAuthPage = ['/login', '/signup', '/forgot-password'].includes(location.pathname);
    const isHomePage = location.pathname === '/';

    if (isAuthenticated && (isAuthPage || isHomePage)) {
      console.log('🔄 Auto-redirecting authenticated user to /upload');
      navigate('/upload', { replace: true });
    }
  }, [token, user, initialized, loading, location.pathname, navigate, isTokenExpired]);

  const login = async (credentials) => {
    try {
      console.log('🔐 Attempting login...');
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: credentials.email || credentials.username,
          password: credentials.password
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Login successful:', { hasToken: !!data.access_token, hasUser: !!data.user });
        
        // Store token and user data
        localStorage.setItem('access_token', data.access_token);
        
        if (data.user) {
          localStorage.setItem('user', JSON.stringify(data.user));
          setUser(data.user);
        }
        
        setToken(data.access_token);

        // Navigation will be handled by the useEffect above
        
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
      console.error('❌ Login error:', error);
      return { success: false, error: 'Network error occurred' };
    } finally {
      setLoading(false);
    }
  };

  const signup = async (userData) => {
    try {
      setLoading(true);
      
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
      console.error('❌ Signup error:', error);
      return { success: false, error: 'Network error occurred' };
    } finally {
      setLoading(false);
    }
  };

  const logout = useCallback(() => {
    console.log('🚪 Logging out...');
    
    // Clear all auth data
    clearAuthStorage();
    setToken(null);
    setUser(null);
    
    // Reset initialization flag to allow re-initialization
    initializationRef.current = false;
    setInitialized(false);
    
    // Redirect to home page
    navigate('/', { replace: true });
    
    console.log('✅ Logout complete');
  }, [clearAuthStorage, navigate]);

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
      console.error('❌ Forgot password error:', error);
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
      console.error('❌ Verify code error:', error);
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
      console.error('❌ Reset password error:', error);
      return { success: false, error: 'Network error occurred' };
    }
  };

  const value = {
    user,
    token,
    loading,
    initialized,
    isAuthenticated: !!token && !!user && !isTokenExpired(token),
    login,
    signup,
    logout,
    forgotPassword,
    verifyResetCode,
    resetPassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};