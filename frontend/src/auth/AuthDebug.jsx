import React, { useState } from 'react';

const AuthDebugComponent = () => {
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);

  const testEndpoint = async (endpoint, requiresAuth = false) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      
      console.log(`🧪 Testing ${endpoint}`);
      console.log(`🔑 Token found: ${!!token}`);
      if (token) {
        console.log(`🔑 Token preview: ${token.substring(0, 20)}...`);
      }

      const headers = {
        'Content-Type': 'application/json',
      };

      if (requiresAuth && token) {
        headers['Authorization'] = `Bearer ${token}`;
        console.log(`📋 Adding auth header: Bearer ${token.substring(0, 20)}...`);
      }

      console.log(`📡 Making request to: http://localhost:8001${endpoint}`);
      console.log(`📋 Headers:`, headers);

      const response = await fetch(`http://localhost:8001${endpoint}`, {
        method: 'GET',
        headers: headers,
      });

      console.log(`📝 Response status: ${response.status}`);
      console.log(`📄 Response headers:`, [...response.headers.entries()]);

      const data = await response.json();
      console.log(`📄 Response data:`, data);

      setResults(prev => ({
        ...prev,
        [endpoint]: {
          status: response.status,
          success: response.ok,
          data: data,
          timestamp: new Date().toISOString()
        }
      }));

    } catch (error) {
      console.error(`🚨 Error testing ${endpoint}:`, error);
      setResults(prev => ({
        ...prev,
        [endpoint]: {
          status: 'error',
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        }
      }));
    } finally {
      setLoading(false);
    }
  };

  const testAuthFlow = async () => {
    setResults({});
    
    // Test 1: Basic connectivity
    await testEndpoint('/test-auth-server');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Test 2: Token inspection
    await testEndpoint('/debug-token');
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Test 3: Auth-protected endpoint
    await testEndpoint('/test-auth', true);
  };

  const checkStoredTokens = () => {
    const localToken = localStorage.getItem('authToken');
    const sessionToken = sessionStorage.getItem('authToken');
    const localUserData = localStorage.getItem('userData');
    const sessionUserData = sessionStorage.getItem('userData');

    return {
      localStorage: {
        token: localToken ? `${localToken.substring(0, 20)}...` : null,
        userData: localUserData ? JSON.parse(localUserData) : null
      },
      sessionStorage: {
        token: sessionToken ? `${sessionToken.substring(0, 20)}...` : null,
        userData: sessionUserData ? JSON.parse(sessionUserData) : null
      }
    };
  };

  const testDirectAuthServer = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
      
      if (!token) {
        alert('No token found! Please log in first.');
        setLoading(false);
        return;
      }

      console.log(`🧪 Testing direct auth server call`);
      const response = await fetch('http://localhost:8000/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      console.log(`📝 Auth server response status: ${response.status}`);
      const data = await response.json();
      console.log(`📄 Auth server response:`, data);

      setResults(prev => ({
        ...prev,
        'direct-auth-server': {
          status: response.status,
          success: response.ok,
          data: data,
          timestamp: new Date().toISOString()
        }
      }));

    } catch (error) {
      console.error(`🚨 Direct auth server test failed:`, error);
      setResults(prev => ({
        ...prev,
        'direct-auth-server': {
          status: 'error',
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        }
      }));
    } finally {
      setLoading(false);
    }
  };

  const storedTokens = checkStoredTokens();

  return (
    <div className="p-6 max-w-4xl mx-auto bg-gray-50 rounded-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Auth Debug Tool</h2>
      
      {/* Token Status */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-3">Current Token Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-gray-700">localStorage:</h4>
            <p className="text-sm text-gray-600">
              Token: {storedTokens.localStorage.token || 'None'}
            </p>
            <p className="text-sm text-gray-600">
              User: {storedTokens.localStorage.userData?.username || 'None'}
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-700">sessionStorage:</h4>
            <p className="text-sm text-gray-600">
              Token: {storedTokens.sessionStorage.token || 'None'}
            </p>
            <p className="text-sm text-gray-600">
              User: {storedTokens.sessionStorage.userData?.username || 'None'}
            </p>
          </div>
        </div>
      </div>

      {/* Test Buttons */}
      <div className="mb-6 space-x-4">
        <button
          onClick={testAuthFlow}
          disabled={loading}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Testing...' : 'Test Full Auth Flow'}
        </button>
        
        <button
          onClick={testDirectAuthServer}
          disabled={loading}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
        >
          Test Direct Auth Server
        </button>
        
        <button
          onClick={() => testEndpoint('/debug-token')}
          disabled={loading}
          className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
        >
          Debug Token Only
        </button>
      </div>

      {/* Results */}
      <div className="space-y-4">
        {Object.entries(results).map(([endpoint, result]) => (
          <div key={endpoint} className="p-4 bg-white rounded-lg shadow">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-gray-800">{endpoint}</h3>
              <span className={`px-2 py-1 rounded text-sm ${
                result.success 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {result.status}
              </span>
            </div>
            
            <div className="text-sm text-gray-600 mb-2">
              {result.timestamp}
            </div>
            
            <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
              {JSON.stringify(result.data || result.error, null, 2)}
            </pre>
          </div>
        ))}
      </div>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
        <h3 className="font-semibold text-yellow-800 mb-2">Debug Instructions:</h3>
        <ol className="text-sm text-yellow-700 space-y-1">
          <li>1. Make sure you're logged in (check token status above)</li>
          <li>2. Click "Test Full Auth Flow" to test all endpoints</li>
          <li>3. Check the browser console for detailed logs</li>
          <li>4. Check your prediction API server logs for auth debugging output</li>
          <li>5. If "Test Direct Auth Server" works but prediction API doesn't, the issue is in the backend-to-backend communication</li>
        </ol>
      </div>
    </div>
  );
};

export default AuthDebugComponent;