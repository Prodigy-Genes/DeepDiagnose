// src/debug/DebugWrapper.jsx
import React, { Suspense } from 'react';
import AuthDebugComponent from '../auth/AuthDebug';

const DebugWrapper = () => (
  <div className="p-4">
    <h1 className="text-2xl font-bold mb-4">Developer Tools</h1>
    <Suspense fallback={<div>Loading debug tools...</div>}>
      <AuthDebugComponent />
    </Suspense>
  </div>
);

export default DebugWrapper;