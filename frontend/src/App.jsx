import React, { useState } from "react";
import "./App.css";
import Home from "./pages/Home/home"
// This component handles the prediction results and displays them to the user
// It uses the useState hook to manage the state of the prediction result
function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="App">
      <Home />
    </div>
  );
}

export default App;
// This is the main application component that combines the image upload and prediction result components