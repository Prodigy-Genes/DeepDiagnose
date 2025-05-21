import React from 'react';
import './GetStarted.css'; 
import '@fortawesome/fontawesome-free/css/all.min.css'; 

const GetStarted = () => {
    const handleClick = () => {
        // Redirect to the image upload page
        const uploadUrl = `${window.location.origin}/upload`;
        window.open(uploadUrl, '_blank', 'noopener,noreferrer' );
    };

    return (
        <div className="cta-button-container">
            <button className="get-started-btn" onClick={handleClick}>
                Get Started
            </button>
        </div>
    );
};

export default GetStarted;