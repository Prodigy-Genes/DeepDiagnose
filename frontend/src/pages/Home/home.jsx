import React from 'react';
import './Home.css';
import Header from '../../components/home-page/header/header';
import Footer from '../../components/home-page/footer/footer';
import Body from './Body';
import { Link } from 'react-router-dom';
import { useAuth } from '../../components/Contexts/AuthContext';
import '../../components/Route/ProtectedRoute.css'; 

export default function Home() {
  const { isAuthenticated, loading, user } = useAuth();

  // Show loading state while checking authentication
if (loading) {
  return (
    <div className="page-container">
      <Header />
      <main className="content">
        <div className="loading-container">
          <div className="loading-spinner-wrapper">
            <div className="loading-spinner"></div>
            <p className="loading-text">Loading...</p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

  // Show welcome message for authenticated users
if (isAuthenticated && user) {
  return (
    <div className="page-container">
      <Header />
      <main className="content">
        <div className="welcome-container">
          <div className="welcome-card">
            <div className="welcome-icon">
              <i className="fas fa-check-circle"></i>
            </div>
            <h2 className="welcome-title">Welcome back!</h2>
            <p className="welcome-message">
              Hello {user.username || user.email}! Redirecting you to the app...
            </p>
            <div className="welcome-spinner"></div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

  // Default home page for non-authenticated users
  return (
    <div className="page-container">
      <Header />
      <main className="content">
        {/* your page's main content goes here */}
        <Body />
      </main>
      <Footer />

      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4">
          <Link 
            to="/debug"
            className="bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-gray-700 transition"
          >
            🛠️ Debug Tools
          </Link>
        </div>
      )}
    </div>
  );
}