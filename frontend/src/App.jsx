// App.jsx
import React from "react";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './components/Contexts/AuthContext';
import ProtectedRoute from './components/Route/ProtectedRoute';
import About from "./pages/About/About";
import Home from "./pages/Home/home";
import APIDocs from "./pages/API/APIDocs";
import UploadImage from "./pages/ImageUpload/UploadImage";
import "./App.css";

// Conditional import for development only
const AuthDebugComponent = React.lazy(() => import("./auth/AuthDebug"));

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/api-docs" element={<APIDocs />} />

          {/* Protected routes */}
          <Route 
            path="/upload" 
            element={
              <ProtectedRoute>
                <UploadImage />
              </ProtectedRoute>
            } 
          />

          {/* Debug route (development only) */}
          {process.env.NODE_ENV === 'development' && (
            <Route 
              path="/debug" 
              element={
                <ProtectedRoute>
                  <React.Suspense fallback={<div>Loading debug tools...</div>}>
                    <AuthDebugComponent />
                  </React.Suspense>
                </ProtectedRoute>
              } 
            />
          )}
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;