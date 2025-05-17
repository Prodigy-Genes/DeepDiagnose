import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import './MiniHeader.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

export default function MiniHeader() {
  const [visible, setVisible] = useState(false);
  
  useEffect(() => {
    // Show mini header once user scrolls past a certain point
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      if (scrollPosition > 300) {
        setVisible(true);
      } else {
        setVisible(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    
    // Clean up event listener
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);
  
  return (
    <div className={`mini-header-container ${visible ? 'visible' : ''}`}>
      <div className="mini-logo-container">
        <h2>deepdiagnose</h2>
      </div>
      
      <nav className="mini-menu-container">
        <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Home</NavLink>
        <NavLink to="/about" className={({ isActive }) => isActive ? 'active' : ''}>About Us</NavLink>
        <NavLink to="/api-docs" className={({ isActive }) => isActive ? 'active' : ''}>API Docs</NavLink>
      </nav>
      
      <div className="mini-social-container">
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
  );
}