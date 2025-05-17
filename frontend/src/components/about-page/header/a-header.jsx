import './a-header.css'; 
import '@fortawesome/fontawesome-free/css/all.min.css'; 
import { NavLink } from 'react-router-dom';
import MiniHeader from './MiniHeader';

// Add onReadAboutUs prop to handle the button click
export default function Header({ onReadAboutUs }) {
  return (
  <>  
    {/* Mini header */}
    <MiniHeader />
    <header className="a-header-container">
      {/* Top row: logo | menu | socials */}
      <div className="a-top-row">
        <div className="a-logo-container">
          <h2>deepdiagnose</h2>
        </div>
        
        <nav className="a-menu-container">
          <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Home</NavLink>
          <NavLink to="/about" className={({ isActive }) => isActive ? 'active' : ''}>About Us</NavLink>
          <NavLink to="/api-docs" className={({ isActive }) => isActive ? 'active' : ''}>API Docs</NavLink>
        </nav>
        
        <div className="a-social-container">
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
      <div className="a-site-title">
        <h1>About Us</h1>
        <div className="a-cta-button-container">
          <button 
            className="read-about-us"
            onClick={onReadAboutUs} // Use the prop as the click handler
          >
            Know us
          </button>
        </div>
      </div>
    </header>
  </>
  );
}