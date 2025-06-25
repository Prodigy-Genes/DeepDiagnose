// App.js
import React from "react";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import About from "./pages/About/About";
import Home from "./pages/Home/home";
import APIDocs from "./pages/API/APIDocs";
import "./App.css";
import UploadImage from "./pages/ImageUpload/UploadImage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/api-docs" element={<APIDocs />} /> 
        {/* render UploadPage at /upload */}
        <Route path="/upload" element={<UploadImage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
