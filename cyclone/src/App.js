import React, { useState } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import './App.css';

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
          latitude: -7.03125,  // Example latitude
          longitude: 31.0529339857, // Example longitude
        },
      });

      setImages(response.data.images);
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  return (
    <div className="container">
      <h1>Cyclone Predictor</h1>
      <div className="date-pickerclass">
        <label>
          Start Date:
          <DatePicker
            selected={startDate}
            onChange={(date) => setStartDate(date)}
            dateFormat="yyyy-MM-dd"
            isClearable
            placeholderText="Select a start date"
          />
        </label>
        <label>
          End Date:
          <DatePicker
            selected={endDate}
            onChange={(date) => setEndDate(date)}
            dateFormat="yyyy-MM-dd"
            isClearable
            placeholderText="Select an end date"
          />
        </label>
      </div>
      <button className="button" onClick={fetchImages}>Fetch Satellite Images</button>
      <div>
        {images.length > 0 && (
          <h2>Satellite Images:</h2>
        )}
        {images.length === 0 && (
          <h2>No Satellite Images</h2>
        )}
        <div className="image-grid">
          {images.map((imageUrl, index) => (
            <img key={index} src={imageUrl} alt={`Satellite view ${index}`} />
          ))}
        </div>
      </div>
      <footer style={{ marginTop: '20px', textAlign: 'center', color: '#555' }}>
        <p>© 2024 Cyclone Predictor. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default App;
