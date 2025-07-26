import React, { useState, useEffect, useRef } from 'react';
import './ForgotPassword.css';

const ForgotPassword = ({ onBackToSignIn, onClose }) => {
    const [currentStep, setCurrentStep] = useState(1); // 1: Email, 2: Verification, 3: Reset Password
    const [formData, setFormData] = useState({
        email: '',
        verificationCode: '',
        newPassword: '',
        confirmPassword: ''
    });
    const [errors, setErrors] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [countdown, setCountdown] = useState(0);
    const [showPasswords, setShowPasswords] = useState({
        new: false,
        confirm: false
    });
    const [isRevealingPassword, setIsRevealingPassword] = useState({
        new: false,
        confirm: false
    });
    
    const emailInputRef = useRef(null);
    const codeInputRefs = useRef([]);
    const newPasswordRef = useRef(null);
    const confirmPasswordRef = useRef(null);
    const typingTimeoutRef = useRef(null);

    // Countdown timer for resend code
    useEffect(() => {
        let timer;
        if (countdown > 0) {
            timer = setTimeout(() => setCountdown(countdown - 1), 1000);
        }
        return () => clearTimeout(timer);
    }, [countdown]);

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

    // Handle verification code input with auto-focus
    const handleCodeInput = (index, value) => {
        if (value.length > 1) return; // Prevent multiple characters
        
        const newCode = formData.verificationCode.split('');
        newCode[index] = value;
        const updatedCode = newCode.join('');
        
        setFormData(prev => ({
            ...prev,
            verificationCode: updatedCode
        }));

        // Auto-focus next input
        if (value && index < 5) {
            codeInputRefs.current[index + 1]?.focus();
        }
        
        // Clear error when user starts typing
        if (errors.verificationCode) {
            setErrors(prev => ({
                ...prev,
                verificationCode: ''
            }));
        }
    };

    // Handle backspace in code inputs
    const handleCodeKeyDown = (index, e) => {
        if (e.key === 'Backspace' && !e.target.value && index > 0) {
            codeInputRefs.current[index - 1]?.focus();
        }
    };

    // Password visibility toggle with matrix effect
    const togglePasswordVisibility = (field) => {
        if (isRevealingPassword[field]) return;
        
        setIsRevealingPassword(prev => ({
            ...prev,
            [field]: true
        }));
        
        const passwordRef = field === 'new' ? newPasswordRef : confirmPasswordRef;
        const passwordContainer = passwordRef.current?.parentElement;
        const overlay = document.createElement('div');
        overlay.className = 'password-reveal-overlay';
        passwordContainer?.appendChild(overlay);
        
        setTimeout(() => {
            setShowPasswords(prev => ({
                ...prev,
                [field]: !prev[field]
            }));
        }, 300);
        
        setTimeout(() => {
            if (passwordContainer && overlay.parentElement) {
                passwordContainer.removeChild(overlay);
            }
            setIsRevealingPassword(prev => ({
                ...prev,
                [field]: false
            }));
        }, 1200);
    };

    // Validation functions
    const validateEmail = () => {
        const newErrors = {};
        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Please enter a valid email address';
        }
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const validateCode = () => {
        const newErrors = {};
        if (formData.verificationCode.length !== 6) {
            newErrors.verificationCode = 'Please enter the complete 6-digit code';
        }
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const validatePasswords = () => {
        const newErrors = {};
        
        if (!formData.newPassword) {
            newErrors.newPassword = 'New password is required';
        } else if (formData.newPassword.length < 8) {
            newErrors.newPassword = 'Password must be at least 8 characters long';
        }
        
        if (!formData.confirmPassword) {
            newErrors.confirmPassword = 'Please confirm your password';
        } else if (formData.newPassword !== formData.confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
        }
        
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    // API calls
    const handleSendResetEmail = async () => {
        if (!validateEmail()) return;
        
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/auth/forgot-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: formData.email
                })
            });

            if (response.ok) {
                setCurrentStep(2);
                setCountdown(60); // 60 second countdown
            } else {
                const errorData = await response.json();
                setErrors({ email: errorData.detail || 'Failed to send reset email' });
            }
        } catch (error) {
            setErrors({ email: 'Network error. Please try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleVerifyCode = async () => {
        if (!validateCode()) return;
        
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/auth/verify-reset-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: formData.email,
                    code: formData.verificationCode
                })
            });

            if (response.ok) {
                setCurrentStep(3);
            } else {
                const errorData = await response.json();
                setErrors({ verificationCode: errorData.detail || 'Invalid verification code' });
            }
        } catch (error) {
            setErrors({ verificationCode: 'Network error. Please try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleResetPassword = async () => {
        if (!validatePasswords()) return;
        
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/auth/reset-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: formData.email,
                    code: formData.verificationCode,
                    new_password: formData.newPassword
                })
            });

            if (response.ok) {
                // Show success message and redirect to sign in
                setTimeout(() => {
                    onBackToSignIn();
                }, 2000);
            } else {
                const errorData = await response.json();
                setErrors({ submit: errorData.detail || 'Failed to reset password' });
            }
        } catch (error) {
            setErrors({ submit: 'Network error. Please try again.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleResendCode = async () => {
        if (countdown > 0) return;
        
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/auth/forgot-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: formData.email
                })
            });

            if (response.ok) {
                setCountdown(60);
                setFormData(prev => ({ ...prev, verificationCode: '' }));
                // Clear all code inputs
                codeInputRefs.current.forEach(ref => {
                    if (ref) ref.value = '';
                });
            }
        } catch (error) {
            console.error('Failed to resend code:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const renderStepIndicator = () => (
        <div className="step-indicator">
            <div className={`step ${currentStep >= 1 ? 'active' : ''} ${currentStep > 1 ? 'completed' : ''}`}>
                <div className="step-number">1</div>
                <span>Email</span>
            </div>
            <div className="step-line"></div>
            <div className={`step ${currentStep >= 2 ? 'active' : ''} ${currentStep > 2 ? 'completed' : ''}`}>
                <div className="step-number">2</div>
                <span>Verify</span>
            </div>
            <div className="step-line"></div>
            <div className={`step ${currentStep >= 3 ? 'active' : ''}`}>
                <div className="step-number">3</div>
                <span>Reset</span>
            </div>
        </div>
    );

    const renderEmailStep = () => (
        <div className="step-content">
            <div className="step-header">
                <i className="fas fa-envelope step-icon"></i>
                <h3>Enter Your Email</h3>
                <p>We'll send you a verification code to reset your password</p>
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
                />
                {errors.email && <span className="error-message">{errors.email}</span>}
            </div>

            <button
                type="button"
                className={`auth-submit ${isLoading ? 'loading' : ''}`}
                onClick={handleSendResetEmail}
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
                        Send Reset Code
                    </>
                )}
            </button>
        </div>
    );

    const renderVerificationStep = () => (
        <div className="step-content">
            <div className="step-header">
                <i className="fas fa-shield-alt step-icon"></i>
                <h3>Verify Your Identity</h3>
                <p>Enter the 6-digit code sent to <strong>{formData.email}</strong></p>
            </div>

            <div className="form-group">
                <label className="form-label">
                    <i className="fas fa-key"></i>
                    Verification Code
                </label>
                <div className="code-input-container">
                    {[0, 1, 2, 3, 4, 5].map((index) => (
                        <input
                            key={index}
                            ref={el => codeInputRefs.current[index] = el}
                            type="text"
                            maxLength="1"
                            className={`code-input ${errors.verificationCode ? 'error' : ''}`}
                            onChange={(e) => handleCodeInput(index, e.target.value)}
                            onKeyDown={(e) => handleCodeKeyDown(index, e)}
                            disabled={isLoading}
                        />
                    ))}
                </div>
                {errors.verificationCode && <span className="error-message">{errors.verificationCode}</span>}
            </div>

            <div className="resend-section">
                <p>Didn't receive the code?</p>
                <button
                    type="button"
                    className={`resend-button ${countdown > 0 ? 'disabled' : ''}`}
                    onClick={handleResendCode}
                    disabled={countdown > 0 || isLoading}
                >
                    {countdown > 0 ? `Resend in ${countdown}s` : 'Resend Code'}
                </button>
            </div>

            <button
                type="button"
                className={`auth-submit ${isLoading ? 'loading' : ''}`}
                onClick={handleVerifyCode}
                disabled={isLoading}
            >
                {isLoading ? (
                    <>
                        <i className="fas fa-spinner fa-spin"></i>
                        Verifying...
                    </>
                ) : (
                    <>
                        <i className="fas fa-check"></i>
                        Verify Code
                    </>
                )}
            </button>
        </div>
    );

    const renderResetStep = () => (
        <div className="step-content">
            <div className="step-header">
                <i className="fas fa-lock step-icon"></i>
                <h3>Create New Password</h3>
                <p>Choose a strong password for your account</p>
            </div>

            <div className="form-group">
                <label htmlFor="newPassword" className="form-label">
                    <i className="fas fa-key"></i>
                    New Password
                </label>
                <div className="password-container">
                    <input
                        ref={newPasswordRef}
                        type={showPasswords.new ? 'text' : 'password'}
                        id="newPassword"
                        name="newPassword"
                        value={formData.newPassword}
                        onChange={handleInputChange}
                        className={`form-input password-input ${errors.newPassword ? 'error' : ''}`}
                        placeholder="Enter new password"
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        className="password-toggle"
                        onClick={() => togglePasswordVisibility('new')}
                        disabled={isLoading || isRevealingPassword.new}
                    >
                        <i className={`fas ${showPasswords.new ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                    </button>
                </div>
                {errors.newPassword && <span className="error-message">{errors.newPassword}</span>}
            </div>

            <div className="form-group">
                <label htmlFor="confirmPassword" className="form-label">
                    <i className="fas fa-check-circle"></i>
                    Confirm Password
                </label>
                <div className="password-container">
                    <input
                        ref={confirmPasswordRef}
                        type={showPasswords.confirm ? 'text' : 'password'}
                        id="confirmPassword"
                        name="confirmPassword"
                        value={formData.confirmPassword}
                        onChange={handleInputChange}
                        className={`form-input password-input ${errors.confirmPassword ? 'error' : ''}`}
                        placeholder="Confirm new password"
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        className="password-toggle"
                        onClick={() => togglePasswordVisibility('confirm')}
                        disabled={isLoading || isRevealingPassword.confirm}
                    >
                        <i className={`fas ${showPasswords.confirm ? 'fa-eye-slash' : 'fa-eye'}`}></i>
                    </button>
                </div>
                {errors.confirmPassword && <span className="error-message">{errors.confirmPassword}</span>}
            </div>

            {errors.submit && (
                <div className="submit-error">
                    <i className="fas fa-exclamation-circle"></i>
                    {errors.submit}
                </div>
            )}

            <button
                type="button"
                className={`auth-submit ${isLoading ? 'loading' : ''}`}
                onClick={handleResetPassword}
                disabled={isLoading}
            >
                {isLoading ? (
                    <>
                        <i className="fas fa-spinner fa-spin"></i>
                        Resetting Password...
                    </>
                ) : (
                    <>
                        <i className="fas fa-check"></i>
                        Reset Password
                    </>
                )}
            </button>
        </div>
    );

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
            <div className="auth-modal forgot-password-modal">
                <div className="auth-header">
                    <h2 className="auth-title">Reset Password</h2>
                    <p className="auth-subtitle">Secure your account with a new password</p>
                    {onClose && (
                        <button className="auth-close" onClick={onClose}>
                            <i className="fas fa-times"></i>
                        </button>
                    )}
                </div>

                {renderStepIndicator()}

                <div className="forgot-password-content">
                    {currentStep === 1 && renderEmailStep()}
                    {currentStep === 2 && renderVerificationStep()}
                    {currentStep === 3 && renderResetStep()}
                </div>

                <div className="auth-footer">
                    <p>Remember your password?</p>
                    <button className="auth-toggle" onClick={onBackToSignIn}>
                        Back to Sign In
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ForgotPassword;