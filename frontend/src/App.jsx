import React, { useState } from "react";
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import About from "./pages/About/About";
import "./App.css";
import Home from "./pages/Home/home"


function App() {
  const [result, setResult] = useState(null);

  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/about' element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
// This is the main application component that combines the image upload and prediction result components