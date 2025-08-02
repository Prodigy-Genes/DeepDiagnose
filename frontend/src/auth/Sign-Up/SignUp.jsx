import React, { useState, useEffect, useRef } from 'react';
import './SignUp.css';

const SignUp = ({ onToggleAuth, onClose }) => {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
    });
    const [errors, setErrors] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [isRevealingPassword, setIsRevealingPassword] = useState(false);
    const [isRevealingConfirmPassword, setIsRevealingConfirmPassword] = useState(false);
    const [agreedToTerms, setAgreedToTerms] = useState(false);
    
    const usernameInputRef = useRef(null);
    const emailInputRef = useRef(null);
    const passwordInputRef = useRef(null);
    const confirmPasswordInputRef = useRef(null);
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
    const togglePasswordVisibility = (field) => {
        const isConfirmField = field === 'confirm';
        const isCurrentlyRevealing = isConfirmField ? isRevealingConfirmPassword : isRevealingPassword;
        
        if (isCurrentlyRevealing) return; // Prevent multiple clicks during animation
        
        if (isConfirmField) {
            setIsRevealingConfirmPassword(true);
        } else {
            setIsRevealingPassword(true);
        }
        
        // Create and add the reveal overlay
        const passwordContainer = isConfirmField 
            ? confirmPasswordInputRef.current?.parentElement
            : passwordInputRef.current?.parentElement;
        const overlay = document.createElement('div');
        overlay.className = 'password-reveal-overlay';
        passwordContainer?.appendChild(overlay);
        
        // Toggle visibility after a short delay for better effect
        setTimeout(() => {
            if (isConfirmField) {
                setShowConfirmPassword(prev => !prev);
            } else {
                setShowPassword(prev => !prev);
            }
        }, 300);
        
        // Remove overlay and reset state after animation
        setTimeout(() => {
            if (passwordContainer && overlay.parentElement) {
                passwordContainer.removeChild(overlay);
            }
            if (isConfirmField) {
                setIsRevealingConfirmPassword(false);
            } else {
                setIsRevealingPassword(false);
            }
        }, 1200);
    };

    const validateForm = () => {
        const newErrors = {};

        if (!formData.username.trim()) {
            newErrors.username = 'Username is required';
        } else if (formData.username.length < 3) {
            newErrors.username = 'Username must be at least 3 characters';
        } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
            newErrors.username = 'Username can only contain letters, numbers, and underscores';
        }

        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Please enter a valid email address';
        }

        if (!formData.password) {
            newErrors.password = 'Password is required';
        } else if (formData.password.length < 8) {
            newErrors.password = 'Password must be at least 8 characters';
        } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
            newErrors.password = 'Password must contain uppercase, lowercase, and number';
        }

        if (!formData.confirmPassword) {
            newErrors.confirmPassword = 'Please confirm your password';
        } else if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
        }

        if (!agreedToTerms) {
            newErrors.terms = 'You must agree to the terms and conditions';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const redirectToUpload = () => {
        // Use same-window navigation instead of opening new tab
        window.location.href = '/upload';
    };

    // Handle form submission - FIXED VERSION
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!validateForm()) {
            return;
        }

        setIsLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/auth/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: formData.username,
                    email: formData.email,
                    password: formData.password
                })
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Signup successful:', result);
                
                // PROBLEM 1 FIX: Signup only returns user_id, not access_token
                // Need to login after successful signup to get token
                console.log('Signup successful, now logging in...');
                
                // Automatically log in the user after successful signup
                const loginResponse = await fetch('http://localhost:8000/auth/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        username: formData.email,  // Use email for login
                        password: formData.password
                    })
                });

                if (loginResponse.ok) {
                    const loginResult = await loginResponse.json();
                    console.log('Auto-login successful:', loginResult);
                    
                    // Store token and user data
                    localStorage.setItem('authToken', loginResult.access_token);
                    if (loginResult.user) {
                        localStorage.setItem('userData', JSON.stringify(loginResult.user));
                    }

                    // Redirect to upload page
                    redirectToUpload();
                    
                    // Close modal
                    onClose && onClose();
                } else {
                    console.error('Auto-login failed after signup');
                    setErrors({ submit: 'Account created but auto-login failed. Please sign in manually.' });
                }
            } else {
                const errorData = await response.json();
                setErrors({ submit: errorData.detail || 'Signup failed. Please try again.' });
            }
        } catch (error) {
            console.error('Signup error:', error);
            setErrors({ submit: 'Network error. Please check your connection and try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleTermsClick = () => {
        // Implement terms and conditions modal/page
        console.log('Terms and conditions clicked');
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
            <div className="auth-modal signup-modal">
                <div className="auth-header">
                    <h2 className="auth-title">Create Account</h2>
                    <p className="auth-subtitle">Join DeepDiagnose and start analyzing medical scans</p>
                    {onClose && (
                        <button className="auth-close" onClick={onClose}>
                            <i className="fas fa-times"></i>
                        </button>
                    )}
                </div>

                <form onSubmit={handleSubmit} className="auth-form">
                    <div className="form-group">
                        <label htmlFor="username" className="form-label">
                            <i className="fas fa-user"></i>
                            Username
                        </label>
                        <input
                            ref={usernameInputRef}
                            type="text"
                            id="username"
                            name="username"
                            value={formData.username}
                            onChange={handleInputChange}
                            className={`form-input ${errors.username ? 'error' : ''}`}
                            placeholder="Choose a unique username"
                            disabled={isLoading}
                            autoComplete="username"
                        />
                        {errors.username && (
                            <span className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {errors.username}
                            </span>
                        )}
                    </div>

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
                            placeholder="Enter your email address"
                            disabled={isLoading}
                            autoComplete="email"
                        />
                        {errors.email && (
                            <span className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {errors.email}
                            </span>
                        )}
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
                                placeholder="Create a secure password"
                                disabled={isLoading}
                                autoComplete="new-password"
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => togglePasswordVisibility('password')}
                                disabled={isLoading || isRevealingPassword}
                                title={showPassword ? 'Hide password' : 'Show password'}
                            >
                                <i className={`fas ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                            </button>
                        </div>
                        {errors.password && (
                            <span className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {errors.password}
                            </span>
                        )}
                        <div className="password-strength">
                            <small>Must contain uppercase, lowercase, and number (min 8 chars)</small>
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="confirmPassword" className="form-label">
                            <i className="fas fa-shield-alt"></i>
                            Confirm Password
                        </label>
                        <div className="password-container">
                            <input
                                ref={confirmPasswordInputRef}
                                type={showConfirmPassword ? 'text' : 'password'}
                                id="confirmPassword"
                                name="confirmPassword"
                                value={formData.confirmPassword}
                                onChange={handleInputChange}
                                className={`form-input password-input ${errors.confirmPassword ? 'error' : ''}`}
                                placeholder="Confirm your password"
                                disabled={isLoading}
                                autoComplete="new-password"
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => togglePasswordVisibility('confirm')}
                                disabled={isLoading || isRevealingConfirmPassword}
                                title={showConfirmPassword ? 'Hide password' : 'Show password'}
                            >
                                <i className={`fas ${showConfirmPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                            </button>
                        </div>
                        {errors.confirmPassword && (
                            <span className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {errors.confirmPassword}
                            </span>
                        )}
                    </div>

                    <div className="form-options">
                        <label className="checkbox-container">
                            <input
                                type="checkbox"
                                checked={agreedToTerms}
                                onChange={(e) => setAgreedToTerms(e.target.checked)}
                                disabled={isLoading}
                            />
                            <span className="checkmark"></span>
                            <span className="checkbox-text">
                                I agree to the{' '}
                                <button
                                    type="button"
                                    className="terms-link"
                                    onClick={handleTermsClick}
                                    disabled={isLoading}
                                >
                                    Terms & Conditions
                                </button>
                            </span>
                        </label>
                        {errors.terms && (
                            <span className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {errors.terms}
                            </span>
                        )}
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
                                Creating Account...
                            </>
                        ) : (
                            <>
                                <i className="fas fa-user-plus"></i>
                                Create Account
                            </>
                        )}
                    </button>
                </form>

                <div className="auth-footer">
                    <p>Already have an account?</p>
                    <button className="auth-toggle" onClick={onToggleAuth}>
                        Sign In
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SignUp;