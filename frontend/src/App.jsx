import React, { useState } from "react";
import ImageUpload from "./components/ImageUpload/ImageUpload";
// This component allows users to upload an image and get a prediction from the server
import PredictionResult from "./components/PredictionResult/PredictionResult";
import "./styles/App.css";

// This component handles the prediction results and displays them to the user
// It uses the useState hook to manage the state of the prediction result
function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="App">
      <h1>DeepDiagnose</h1>
      <ImageUpload onResult={setResult} />
      <PredictionResult result={result} />
    </div>
  );
}

export default App;
// This is the main application component that combines the image upload and prediction result components