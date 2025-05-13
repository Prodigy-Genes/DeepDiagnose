import { useState, useEffect } from 'react';
import './header.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

export default function Header() {
  const phrases = [
    'Your AI-Powered X-Ray Diagnosis',
    'Instant Results, Expert Insights',
    'Detect anomalies in seconds',
    'Empowering Radiologists Everywhere',
  ];

  const [index, setIndex] = useState(0);      // which phrase
  const [subIndex, setSubIndex] = useState(0); // how many letters
  const [blink, setBlink] = useState(true);   // cursor blink
  const [reverse, setReverse] = useState(false);

  // Type / delete effect
  useEffect(() => {
    if (index >= phrases.length) return;

    if (!reverse && subIndex === phrases[index].length + 1) {
      // pause at full phrase
      setReverse(true);
      return;
    }
    if (reverse && subIndex === 0) {
      setReverse(false);
      setIndex((prev) => (prev + 1) % phrases.length);
      return;
    }

    const timeout = setTimeout(() => {
      setSubIndex((prev) => prev + (reverse ? -1 : 1));
    }, reverse ? 50 : 150);

    return () => clearTimeout(timeout);
  }, [subIndex, index, reverse]);

  // Cursor blink
  useEffect(() => {
    const blinkInterval = setInterval(() => {
      setBlink((prev) => !prev);
    }, 500);
    return () => clearInterval(blinkInterval);
  }, []);

  return (
    <header className="header-container">
      {/* Video Background */}
      <div className="video-background">
        <video autoPlay muted loop playsInline className="background-video">
          <source src="/videos/header_video.mp4"/>
          Your browser does not support the video tag.
        </video>
        <div className="overlay"></div>
      </div>
      
      {/* Top row: logo | menu | socials */}
      <div className="top-row">
        <div className="logo-container">
          <h2>deepdiagnose</h2>
        </div>

        <nav className="menu-container">
          <a href="#">Home</a>
          <a href="#">About</a>
          <a href="#">Services</a>
          <a href="#">Contact Us</a>
        </nav>

        <div className="social-container">
          <a href="#"><i className="fa-brands fa-facebook"></i></a>
          <a href="#"><i className="fa-brands fa-instagram"></i></a>
          <a href="#"><i className="fa-brands fa-twitter"></i></a>
          <a href="#"><i className="fa-brands fa-linkedin"></i></a>
        </div>

      </div>

      {/* Site title below */}
      <div className="site-title">
        <h1>deepdiagnose</h1>
        <h3>{`${phrases[index].substring(0, subIndex)}${blink ? '|' : ' '}`}</h3>
        <div className="cta-button-container">
          <button className="get-started-btn">Get Started</button>
        </div>
      </div>
      
    </header>
  );
}