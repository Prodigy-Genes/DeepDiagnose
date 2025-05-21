// src/AppRoutes.jsx
import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Lazy‐load your page components
const Home    = lazy(() => import('./pages/Home/home'));
const About   = lazy(() => import('./pages/About/About'));


export default function AppRoutes() {
  return (
    <Suspense fallback={<div>Loading…</div>}>
      <Routes>
        <Route path="/"            element={<Home />} />
        <Route path="/about"       element={<About />} />
        
        {/* Redirect any unknown URL to /404 */}
        <Route path="*"            element={<Navigate to="/404" replace />} />
      </Routes>
    </Suspense>
  );
}
