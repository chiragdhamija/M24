import React, { useState } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import './App.css';
import ImageSlideshow from './ImageComparisonSlider';

const App = () => {
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [images, setImages] = useState([]);

  const fetchImages = async () => {
    if (!startDate || !endDate) {
      alert("Please select both start and end dates.");
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
          <div className="flex flex-col gap-2">
            <label className="font-medium">
              Start Date:
              <DatePicker
                selected={startDate}
                onChange={(date) => setStartDate(date)}
                dateFormat="yyyy-MM-dd"
                isClearable
                placeholderText="Select a start date"
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </label>
          </div>
          
          <div className="flex flex-col gap-2">
            <label className="font-medium">
              End Date:
              <DatePicker
                selected={endDate}
                onChange={(date) => setEndDate(date)}
                dateFormat="yyyy-MM-dd"
                isClearable
                placeholderText="Select an end date"
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </label>
          </div>

          <button
            onClick={fetchImages}
            className="bg-gradient-to-r from-blue-600 to-blue-500 text-white py-3 px-4 rounded-lg hover:from-blue-700 hover:to-blue-600 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
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