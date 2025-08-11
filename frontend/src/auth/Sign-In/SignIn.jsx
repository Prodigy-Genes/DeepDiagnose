import React, { useState, useEffect, useRef } from 'react';
import './SignIn.css';
import ForgotPassword from '../ForgotPassword/ForgotPassword';
import { useAuth } from '../../components/Contexts/AuthContext'

const SignIn = ({ onToggleAuth, onClose }) => {
    const [formData, setFormData] = useState({
        email: '',
        password: ''
    });
    const [errors, setErrors] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [rememberMe, setRememberMe] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [isRevealingPassword, setIsRevealingPassword] = useState(false);
    const [showForgotPassword, setShowForgotPassword] = useState(false);
    
    const emailInputRef = useRef(null);
    const passwordInputRef = useRef(null);
    const typingTimeoutRef = useRef(null);

    // Get auth functions from context
    const { login, isAuthenticated } = useAuth();

    // If user is already authenticated, close modal
    useEffect(() => {
        if (isAuthenticated) {
            onClose && onClose();
        }
    }, [isAuthenticated, onClose]);

    // Add typing animation effect
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
        
        // Clear error when user starts typing
        if (errors[name]) {
            setErrors(prev => ({
                ...prev,
                [name]: ''
            }));
        }

        // Add typing animation class
        const inputElement = e.target;
        inputElement.classList.add('typing');
        
        // Clear previous timeout
        if (typingTimeoutRef.current) {
            clearTimeout(typingTimeoutRef.current);
        }
        
        // Remove typing class after animation
        typingTimeoutRef.current = setTimeout(() => {
            inputElement.classList.remove('typing');
        }, 800);
    };

    // Handle password visibility toggle with reveal animation
    const togglePasswordVisibility = () => {
        if (isRevealingPassword) return; // Prevent multiple clicks during animation
        
        setIsRevealingPassword(true);
        
        // Create and add the reveal overlay
        const passwordContainer = passwordInputRef.current?.parentElement;
        const overlay = document.createElement('div');
        overlay.className = 'password-reveal-overlay';
        passwordContainer?.appendChild(overlay);
        
        // Toggle visibility after a short delay for better effect
        setTimeout(() => {
            setShowPassword(prev => !prev);
        }, 300);
        
        // Remove overlay and reset state after animation
        setTimeout(() => {
            if (passwordContainer && overlay.parentElement) {
                passwordContainer.removeChild(overlay);
            }
            setIsRevealingPassword(false);
        }, 1200);
    };

    const validateForm = () => {
        const newErrors = {};

        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Please enter a valid email address';
        }

        if (!formData.password) {
            newErrors.password = 'Password is required';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!validateForm()) {
            return;
        }

        setIsLoading(true);
        
        try {
            // Use AuthContext login function
            const result = await login({
                email: formData.email, // Use email field directly
                password: formData.password
            });

            if (result.success) {
                console.log('Signin successful:', result.data);
                
                // Handle rememberMe - this is now handled by AuthContext
                // but we can still store preference for future logins
                if (rememberMe) {
                    localStorage.setItem('rememberMe', 'true');
                } else {
                    localStorage.removeItem('rememberMe');
                }
                
                // Close modal - routing is handled by AuthContext
                onClose && onClose();
                
            } else {
                // Handle login errors
                setErrors({ submit: result.error });
            }
        } catch (error) {
            console.error('Signin error:', error);
            setErrors({ submit: 'Network error. Please check your connection and try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleForgotPassword = () => {
        // Implement forgot password functionality
        console.log('Forgot password clicked');
        setShowForgotPassword(true);
    };

    const handleBackToSignIn = () => {
        setShowForgotPassword(false);
        // Clear any existing errors when going back
        setErrors({});
    };

    // Cleanup timeout on component unmount
    useEffect(() => {
        return () => {
            if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
            }
        };
    }, []);

    // Load remember me preference
    useEffect(() => {
        const savedRememberMe = localStorage.getItem('rememberMe') === 'true';
        setRememberMe(savedRememberMe);
    }, []);

     // Render forgot password component if requested
    if (showForgotPassword) {
        return (
            <ForgotPassword 
                onBackToSignIn={handleBackToSignIn}
                onClose={onClose}
            />
        );
    }

    return (
        <div className="auth-overlay">
            <div className="auth-modal signin-modal">
                <div className="auth-header">
                    <h2 className="auth-title">Welcome Back</h2>
                    <p className="auth-subtitle">Sign in to access your DeepDiagnose account</p>
                    {onClose && (
                        <button className="auth-close" onClick={onClose}>
                            <i className="fas fa-times"></i>
                        </button>
                    )}
                </div>

                <form onSubmit={handleSubmit} className="auth-form">
                    <div className="form-group">
                        <label htmlFor="email" className="form-label">
                            <i className="fas fa-envelope"></i>
                            Email Address
                        </label>
                        <input
                            ref={emailInputRef}
                            type="email"
                            id="email"
                            name="email"
                            value={formData.email}
                            onChange={handleInputChange}
                            className={`form-input ${errors.email ? 'error' : ''}`}
                            placeholder="Enter your email"
                            disabled={isLoading}
                        />
                        {errors.email && <span className="error-message">{errors.email}</span>}
                    </div>

                    <div className="form-group">
                        <label htmlFor="password" className="form-label">
                            <i className="fas fa-lock"></i>
                            Password
                        </label>
                        <div className="password-container">
                            <input
                                ref={passwordInputRef}
                                type={showPassword ? 'text' : 'password'}
                                id="password"
                                name="password"
                                value={formData.password}
                                onChange={handleInputChange}
                                className={`form-input password-input ${errors.password ? 'error' : ''}`}
                                placeholder="Enter your password"
                                disabled={isLoading}
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={togglePasswordVisibility}
                                disabled={isLoading || isRevealingPassword}
                                title={showPassword ? 'Hide password' : 'Show password'}
                            >
                                <i className={`fas ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                            </button>
                        </div>
                        {errors.password && <span className="error-message">{errors.password}</span>}
                    </div>

                    <div className="form-options">
                        <label className="checkbox-container">
                            <input
                                type="checkbox"
                                checked={rememberMe}
                                onChange={(e) => setRememberMe(e.target.checked)}
                                disabled={isLoading}
                            />
                            <span className="checkmark"></span>
                            <span className="checkbox-text">Remember me</span>
                        </label>
                        
                        <button
                            type="button"
                            className="forgot-password"
                            onClick={handleForgotPassword}
                            disabled={isLoading}
                        >
                            Forgot Password?
                        </button>
                    </div>

                    {errors.submit && (
                        <div className="submit-error">
                            <i className="fas fa-exclamation-circle"></i>
                            {errors.submit}
                        </div>
                    )}

                    <button
                        type="submit"
                        className={`auth-submit ${isLoading ? 'loading' : ''}`}
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <>
                                <i className="fas fa-spinner fa-spin"></i>
                                Signing In...
                            </>
                        ) : (
                            <>
                                <i className="fas fa-sign-in-alt"></i>
                                Sign In
                            </>
                        )}
                    </button>
                </form>

                <div className="auth-footer">
                    <p>Don't have an account?</p>
                    <button className="auth-toggle" onClick={onToggleAuth}>
                        Sign Up
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SignIn;