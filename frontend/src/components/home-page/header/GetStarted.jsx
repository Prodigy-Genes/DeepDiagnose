import React, { useState } from 'react';
import SignIn from '../../../auth/Sign-In/SignIn';
import SignUp from '../../../auth/Sign-Up/SignUp';
import './GetStarted.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

const GetStarted = () => {
    const [showAuth, setShowAuth] = useState(false);
    const [authMode, setAuthMode] = useState('signup'); // 'signin' or 'signup'

    const handleClick = () => {
        // Check if user is authenticated
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        
        if (token) {
            // User is authenticated, redirect to upload page
            const uploadUrl = `${window.location.origin}/upload`;
            window.open(uploadUrl, '_blank', 'noopener,noreferrer');
        } else {
            // User is not authenticated, show signup modal
            setAuthMode('signup');
            setShowAuth(true);
        }
    };

    const handleToggleAuth = () => {
        setAuthMode(authMode === 'signin' ? 'signup' : 'signin');
    };

    const handleCloseAuth = () => {
        setShowAuth(false);
    };

    return (
        <>
            <div className="cta-button-container">
                <button className="get-started-btn" onClick={handleClick}>
                    Get Started
                </button>
            </div>
            
            {showAuth && (
                <>
                    {authMode === 'signin' ? (
                        <SignIn 
                            onToggleAuth={handleToggleAuth}
                            onClose={handleCloseAuth}
                        />
                    ) : (
                        <SignUp 
                            onToggleAuth={handleToggleAuth}
                            onClose={handleCloseAuth}
                        />
                    )}
                </>
            )}
        </>
    );
};

export default GetStarted;