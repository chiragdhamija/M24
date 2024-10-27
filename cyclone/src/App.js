import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';
import ImageSlideshow from './ImageComparisonSlider';

const App = () => {
  const [images, setImages] = useState([]);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [currentDate, setCurrentDate] = useState('');

  // Calculate the start and end dates
  useEffect(() => {
    const today = new Date();
    const tenDaysAgo = new Date(today);
    tenDaysAgo.setDate(today.getDate() - 10);

    setStartDate(tenDaysAgo);
    setEndDate(today);
    setCurrentDate(today.toISOString().split("T")[0]); // Format current date
  }, []);

  const fetchImages = async () => {
    if (!startDate || !endDate) {
      alert("Date calculation error. Please try again.");
      return;
    }

    try {
      const response = await axios.get('http://localhost:5000/google-earth-api', {
        params: {
          startDate: startDate.toISOString().split("T")[0],
          endDate: endDate.toISOString().split("T")[0],
          latitude: -7.03125,
          longitude: 31.0529339857,
        },
      });
      setImages(response.data.images);
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="header-container">
        <div className="title-container">
          <h1 className="main-title">CRAPS</h1>
          <h2 className="subtitle">Cyclone Rapid Predictor System</h2>
        </div>
        <div className="wave"></div>
      </div>

      <div className="container mx-auto px-4">
        <div className="max-w-md mx-auto mb-8 flex flex-col gap-4 bg-white p-6 rounded-lg shadow-lg">
          {/* Display current date */}
          <p className="text-center text-lg font-semibold">Current Date: {currentDate}</p>
          <button
            onClick={fetchImages}
            className="button"
          >
            Fetch Satellite Images
          </button>
        </div>

        {images.length > 0 && (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold text-center mb-4">Satellite Images</h2>
            <ImageSlideshow
              images={images}
              startDate={startDate}
              endDate={endDate}
            />
          </div>
        )}

        <footer className="mt-8 text-center text-gray-500 py-6">
          <p>© 2024 CRAPS - Cyclone Rapid Predictor System. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
