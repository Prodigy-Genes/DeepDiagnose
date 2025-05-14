// src/pages/About.jsx
import React, { useState } from 'react';
import Header from '../../components/about-page/header/a-header';
import DevelopersInfo from '../../components/about-page/developers_info/developers_info'; // Import the DevelopersInfo component
import './About.css'; // Import the About page CSS

export default function About() {
  const [showDevelopersInfo, setShowDevelopersInfo] = useState(false);
  
  return (
    <div className="about-page">
      <Header onReadAboutUs={() => setShowDevelopersInfo(true)} />
      <main className="about-content">
        <section className="about-section">
          <div className="container">
            <h2>Our Story</h2>
            <p>
              DeepDiagnose was founded with a clear mission: to revolutionize
              medical imaging through artificial intelligence, enhancing early
              disease detection and improving patient care. Recognizing the
              challenges of manual image analysis—its time-consuming nature and
              susceptibility to human error—a team of AI researchers and
              healthcare professionals came together to develop an advanced
              AI-powered imaging system.
            </p>
            <p>
              Using deep learning techniques, particularly Convolutional Neural
              Networks (CNNs), DeepDiagnose intelligently classifies and
              segments medical images, providing healthcare providers with
              precise and efficient diagnostic support. Designed as a
              cloud-based web application, the platform ensures scalability,
              accessibility, and streamlined medical image analysis, empowering
              clinicians and patients with faster and more accurate diagnoses.
            </p>
            <p>
              This initiative aligns with Sustainable Development Goal (SDG) 3,
              reinforcing the global commitment to ensuring good health and
              well-being through early disease detection and improved
              diagnostic precision.
            </p>

            <h3>Our Mission</h3>
            <p>
              We are committed to developing cutting-edge AI solutions that
              support healthcare providers in making faster, more accurate
              diagnoses, ultimately saving lives and reducing healthcare costs.
            </p>

            <h3>Our Team</h3>
            <p>
              Our multidisciplinary team brings together expertise in
              artificial intelligence, software engineering, and data science.
              This unique combination of skills allows us to develop solutions
              that are both technically advanced and clinically relevant.
            </p>

            <div className="team-button-container">
              <button
                className="read-about-us"
                onClick={() => setShowDevelopersInfo(true)}
              >
                Meet Our Team
              </button>
            </div>
          </div>
        </section>
      </main>

      <DevelopersInfo
        isOpen={showDevelopersInfo}
        onClose={() => setShowDevelopersInfo(false)}
      />
    </div>
  );
}