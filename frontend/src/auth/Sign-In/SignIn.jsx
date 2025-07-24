import React, { useState, useEffect, useRef } from 'react';
import './SignIn.css';

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
    
    const emailInputRef = useRef(null);
    const passwordInputRef = useRef(null);
    const typingTimeoutRef = useRef(null);

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

    const redirectToUpload = () => {
        // Instead of opening new tab, navigate in same window
        window.location.href = '/upload';
    };

    const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
        return;
    }

    setIsLoading(true);
    
    try {
        const response = await fetch('http://localhost:8000/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: formData.email,  // Backend expects 'username' field
                password: formData.password
            })
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Signin successful:', result);
            
            // Store token based on rememberMe preference
            const storage = rememberMe ? localStorage : sessionStorage;
            storage.setItem('authToken', result.access_token);
            
            // Store user data (now included in login response)
            if (result.user) {
                storage.setItem('userData', JSON.stringify(result.user));
                console.log('User data stored:', result.user);
            }

            // Give a moment for storage to complete, then redirect
            setTimeout(() => {
                redirectToUpload();
            }, 100);
            
            // Close modal if provided
            onClose && onClose();
            
        } else {
            const errorData = await response.json();
            
            // Handle array of validation errors
            if (Array.isArray(errorData.detail)) {
                const errorMessages = errorData.detail.map(err => err.msg).join('. ');
                setErrors({ submit: errorMessages });
            } 
            // Handle single error message
            else if (errorData.detail) {
                setErrors({ submit: errorData.detail });
            } 
            // Default error
            else {
                setErrors({ submit: 'Invalid credentials. Please try again.' });
            }
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
    };

    // Cleanup timeout on component unmount
    useEffect(() => {
        return () => {
            if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
            }
        };
    }, []);

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