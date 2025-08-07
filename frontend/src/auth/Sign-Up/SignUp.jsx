import React, { useState, useEffect, useRef } from 'react';
import './SignUp.css';

const SignUp = ({ onToggleAuth, onClose }) => {
    // Form states
    const [currentStep, setCurrentStep] = useState(1); // 1 = signup form, 2 = OTP verification
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
    });
    const [otpData, setOtpData] = useState({
        otp: '',
        email: '' // Will be set from formData.email
    });
    
    // UI states
    const [errors, setErrors] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [isRevealingPassword, setIsRevealingPassword] = useState(false);
    const [isRevealingConfirmPassword, setIsRevealingConfirmPassword] = useState(false);
    const [agreedToTerms, setAgreedToTerms] = useState(false);
    const [otpTimer, setOtpTimer] = useState(0);
    const [canResendOtp, setCanResendOtp] = useState(false);
    
    // Refs
    const usernameInputRef = useRef(null);
    const emailInputRef = useRef(null);
    const passwordInputRef = useRef(null);
    const confirmPasswordInputRef = useRef(null);
    const otpInputRefs = useRef([]);
    const typingTimeoutRef = useRef(null);
    const timerRef = useRef(null);

    // OTP Timer Effect
    useEffect(() => {
        if (currentStep === 2 && otpTimer > 0) {
            timerRef.current = setInterval(() => {
                setOtpTimer(prev => {
                    if (prev <= 1) {
                        setCanResendOtp(true);
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
        
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, [currentStep, otpTimer]);

    // Format timer display
    const formatTimer = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

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

    // Handle OTP input change
    const handleOtpChange = (index, value) => {
        // Only allow digits
        if (!/^\d*$/.test(value)) return;
        
        const newOtp = otpData.otp.split('');
        newOtp[index] = value;
        
        setOtpData(prev => ({
            ...prev,
            otp: newOtp.join('')
        }));
        
        // Auto-focus next input
        if (value && index < 5) {
            otpInputRefs.current[index + 1]?.focus();
        }
        
        // Clear errors
        if (errors.otp) {
            setErrors(prev => ({ ...prev, otp: '' }));
        }
    };

    // Handle OTP input keydown
    const handleOtpKeyDown = (index, e) => {
        if (e.key === 'Backspace' && !otpData.otp[index] && index > 0) {
            otpInputRefs.current[index - 1]?.focus();
        }
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

    // Validate signup form
    const validateSignupForm = () => {
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

    // Validate OTP
    const validateOtp = () => {
        const newErrors = {};
        
        if (!otpData.otp || otpData.otp.length !== 6) {
            newErrors.otp = 'Please enter the complete 6-digit code';
        }
        
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const redirectToUpload = () => {
        // Use same-window navigation instead of opening new tab
        window.location.href = '/upload';
    };

    // Handle signup form submission (Step 1)
    const handleSignupSubmit = async (e) => {
        e.preventDefault();
        
        if (!validateSignupForm()) {
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
                console.log('OTP sent successfully:', result);
                
                // Move to OTP verification step
                setOtpData(prev => ({ ...prev, email: formData.email }));
                setCurrentStep(2);
                
                // Start OTP timer (10 minutes = 600 seconds)
                setOtpTimer(600);
                setCanResendOtp(false);
                
                // Focus first OTP input
                setTimeout(() => {
                    otpInputRefs.current[0]?.focus();
                }, 100);
                
            } else {
                const errorData = await response.json();
                setErrors({ submit: errorData.detail || 'Failed to send verification code. Please try again.' });
            }
        } catch (error) {
            console.error('Signup error:', error);
            setErrors({ submit: 'Network error. Please check your connection and try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    // Handle OTP verification submission (Step 2)
    const handleOtpSubmit = async (e) => {
        e.preventDefault();
        
        if (!validateOtp()) {
            return;
        }

        setIsLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/auth/verify-otp', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: otpData.email,
                    otp: otpData.otp
                })
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Account created successfully:', result);
                
                // Store token and user data (user is automatically logged in)
                localStorage.setItem('authToken', result.access_token);
                if (result.user) {
                    localStorage.setItem('userData', JSON.stringify(result.user));
                }

                // Redirect to upload page
                redirectToUpload();
                
                // Close modal
                onClose && onClose();
                
            } else {
                const errorData = await response.json();
                setErrors({ otp: errorData.detail || 'Invalid verification code. Please try again.' });
            }
        } catch (error) {
            console.error('OTP verification error:', error);
            setErrors({ otp: 'Network error. Please check your connection and try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    // Handle resend OTP
    const handleResendOtp = async () => {
        if (!canResendOtp || isLoading) return;
        
        setIsLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/auth/resend-otp', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: otpData.email
                })
            });

            if (response.ok) {
                // Reset timer and clear OTP
                setOtpTimer(600);
                setCanResendOtp(false);
                setOtpData(prev => ({ ...prev, otp: '' }));
                setErrors({});
                
                // Focus first input
                otpInputRefs.current[0]?.focus();
                
            } else {
                const errorData = await response.json();
                setErrors({ otp: errorData.detail || 'Failed to resend code. Please try again.' });
            }
        } catch (error) {
            console.error('Resend OTP error:', error);
            setErrors({ otp: 'Network error. Please try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    // Go back to signup form
    const goBackToSignup = () => {
        setCurrentStep(1);
        setOtpData({ otp: '', email: '' });
        setOtpTimer(0);
        setCanResendOtp(false);
        setErrors({});
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
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, []);

    return (
        <div className="auth-overlay">
            <div className="auth-modal signup-modal">
                <div className="auth-header">
                    <h2 className="auth-title">
                        {currentStep === 1 ? 'Create Account' : 'Verify Your Email'}
                    </h2>
                    <p className="auth-subtitle">
                        {currentStep === 1 
                            ? 'Join DeepDiagnose and start analyzing medical scans'
                            : `We sent a 6-digit code to ${otpData.email}`
                        }
                    </p>
                    {onClose && (
                        <button className="auth-close" onClick={onClose}>
                            <i className="fas fa-times"></i>
                        </button>
                    )}
                </div>

                {currentStep === 1 ? (
                    // STEP 1: Signup Form
                    <form onSubmit={handleSignupSubmit} className="auth-form">
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
                                    Sending Code...
                                </>
                            ) : (
                                <>
                                    <i className="fas fa-paper-plane"></i>
                                    Send Verification Code
                                </>
                            )}
                        </button>
                    </form>
                ) : (
                    // STEP 2: OTP Verification Form
                    <form onSubmit={handleOtpSubmit} className="auth-form otp-form">
                        <div className="otp-info">
                            <div className="otp-icon">
                                <i className="fas fa-envelope-open"></i>
                            </div>
                            <p>Check your email for a 6-digit verification code</p>
                            {otpTimer > 0 && (
                                <div className="otp-timer">
                                    <i className="fas fa-clock"></i>
                                    Code expires in {formatTimer(otpTimer)}
                                </div>
                            )}
                        </div>

                        <div className="form-group">
                            <label className="form-label otp-label">
                                <i className="fas fa-key"></i>
                                Verification Code
                            </label>
                            <div className="otp-inputs">
                                {Array.from({ length: 6 }).map((_, index) => (
                                    <input
                                        key={index}
                                        ref={el => otpInputRefs.current[index] = el}
                                        type="text"
                                        inputMode="numeric"
                                        pattern="[0-9]*"
                                        maxLength="1"
                                        value={otpData.otp[index] || ''}
                                        onChange={(e) => handleOtpChange(index, e.target.value)}
                                        onKeyDown={(e) => handleOtpKeyDown(index, e)}
                                        className={`otp-input ${errors.otp ? 'error' : ''}`}
                                        disabled={isLoading}
                                        autoComplete="one-time-code"
                                    />
                                ))}
                            </div>
                            {errors.otp && (
                                <span className="error-message">
                                    <i className="fas fa-exclamation-circle"></i>
                                    {errors.otp}
                                </span>
                            )}
                        </div>

                        <div className="otp-actions">
                            <button
                                type="button"
                                className={`resend-btn ${canResendOtp && !isLoading ? 'active' : 'disabled'}`}
                                onClick={handleResendOtp}
                                disabled={!canResendOtp || isLoading}
                            >
                                {isLoading ? (
                                    <>
                                        <i className="fas fa-spinner fa-spin"></i>
                                        Sending...
                                    </>
                                ) : (
                                    <>
                                        <i className="fas fa-redo"></i>
                                        {canResendOtp ? 'Resend Code' : `Wait ${formatTimer(otpTimer)}`}
                                    </>
                                )}
                            </button>

                            <button
                                type="button"
                                className="back-btn"
                                onClick={goBackToSignup}
                                disabled={isLoading}
                            >
                                <i className="fas fa-arrow-left"></i>
                                Change Email
                            </button>
                        </div>

                        <button
                            type="submit"
                            className={`auth-submit ${isLoading ? 'loading' : ''}`}
                            disabled={isLoading || otpData.otp.length !== 6}
                        >
                            {isLoading ? (
                                <>
                                    <i className="fas fa-spinner fa-spin"></i>
                                    Verifying...
                                </>
                            ) : (
                                <>
                                    <i className="fas fa-check-circle"></i>
                                    Verify & Create Account
                                </>
                            )}
                        </button>
                    </form>
                )}

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