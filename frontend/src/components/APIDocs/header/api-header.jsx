import './api-header.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import { NavLink } from 'react-router-dom';
import MiniHeader from '../../about-page/header/MiniHeader';

// Add onAPIDocs prop to handle the button click
export default function Header({ onAPIDocs }) {
    return (
        <>
        {/* Mini header */}
        <MiniHeader />
        <header className="api-header-container">
            {/* Top row: logo | menu | socials */}
            <div className="api-top-row">
            <div className="api-logo-container">
                <h2>deepdiagnose</h2>
            </div>
            
            <nav className="api-menu-container">
                <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Home</NavLink>
                <NavLink to="/about" className={({ isActive }) => isActive ? 'active' : ''}>About Us</NavLink>
                <NavLink to="/api-docs" className={({ isActive }) => isActive ? 'active' : ''}>API Docs</NavLink>
            </nav>
            
            <div className="api-social-container">
                <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-facebook"></i>
                </a>
                <a href="https://instagram.com" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-instagram"></i>
                </a>
                <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-twitter"></i>
                </a>
                <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                <i className="fa-brands fa-linkedin"></i>
                </a>
            </div>
            </div>
            
            {/* Site title */}
            <div className="api-site-title">
            <h1>API Documentation</h1>
            
            </div>
        </header>
        </>
    );
    }