// src/components/DevelopersInfo.jsx
import React, { useState } from 'react';
import './developers_info.css';

// Developer data - you can expand this array with more team members
const developers = [
    {
        id: 1,
        name: "Owuraku (Amponsah) Oduro ",
        title: "AWS Certified Cloud Practitioner & Java Developer",
        image: "https://media.licdn.com/dms/image/v2/D4D35AQGum8pNQ3MsmA/profile-framedphoto-shrink_400_400/profile-framedphoto-shrink_400_400/0/1737660719709?e=1747832400&v=beta&t=xVaEPCz76SLRfUnQiGcOtXRmw-sta7r9nUjwO75r8oc", // Online image URL
        linkedin: "https://www.linkedin.com/in/owuraku-oduro-875b91238/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B%2B14qJjCXQgiQh95SlmnTMw%3D%3D",
        description: "A proactive individual dedicated to positively impacting society, sparking discussions about and contributing to innovation and technology drives.",
        specialty: "Database Management, Cloud Computing, Human-Computer Interaction, Networking"
    },
    {
        id: 2,
        name: "Osei Joseph (prodigygenes) Aboagye ",
        title: "Software Developer | Artist ",
        image: "https://media.licdn.com/dms/image/v2/C4E03AQGt-tlT-AUR-A/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1660741116356?e=1752710400&v=beta&t=4rOa7AZOaudcmLKX8OMN2ddF5q2u7yTAjDsf1ui1kqY", // Online image URL
        linkedin: "https://www.linkedin.com/in/osei-joseph-aboagye-2a3a13238/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3ByjpbCATlQ3yt7fkzZJWjZg%3D%3D",
        description: "A passionate software developer with a keen interest in AI and machine learning. I enjoy creating innovative solutions that enhance user experiences.",
        specialty: "Frontend Development, Backend Development, AI/ML"
    },
    
];

export default function DevelopersInfo({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState(1);
  
  if (!isOpen) return null;
  
  return (
    <div className="modal-overlay">
      <div className="developers-modal">
        <button className="close-button" onClick={onClose}>
          <i className="fas fa-times"></i>
        </button>
        
        <h2 className="modal-title">Our Team</h2>
        
        <div className="tabs-container">
          {developers.map(dev => (
            <button 
              key={dev.id}
              className={`tab-button ${activeTab === dev.id ? 'active' : ''}`}
              onClick={() => setActiveTab(dev.id)}
            >
              {dev.name}
            </button>
          ))}
        </div>
        
        <div className="cards-container">
          {developers.map(dev => (
            <div 
              key={dev.id} 
              className={`developer-card ${activeTab === dev.id ? 'active' : ''}`}
            >
              <div className="card-header">
                <div className="dev-image">
                  <img src={dev.image} alt={dev.name} onError={(e) => {
                    e.target.onerror = null; 
                    e.target.src = "https://via.placeholder.com/150?text=Developer";
                  }} />
                </div>
                <div className="dev-info">
                  <h3>{dev.name}</h3>
                  <h4>{dev.title}</h4>
                  <div className="specialty-tags">
                    {dev.specialty.split(',').map((specialty, index) => (
                      <span key={index} className="specialty-tag">{specialty.trim()}</span>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="card-body">
                <p>{dev.description}</p>
                <a 
                  href={dev.linkedin} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="linkedin-link"
                >
                  <i className="fab fa-linkedin"></i> View LinkedIn Profile
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}