import React from 'react';
import './footer.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

export default function Footer() {
  return (
    <footer className="footer-container">
      
      <div className="footer-copy">
        &copy; {new Date().getFullYear()} DeepDiagnose. All rights reserved.
      </div>
    </footer>
  );
}