import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Calendar } from 'lucide-react';
import './image.css';

const ImageSlideshow = ({ images, startDate, endDate }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    setIsLoading(true);
    setImageLoaded(false);
  }, [currentIndex]);

  const goToNextSlide = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === images.length - 1 ? 0 : prevIndex + 1
    );
  };

  const goToPreviousSlide = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === 0 ? images.length - 1 : prevIndex - 1
    );
  };

  const getDaysBetweenDates = (start, end) => {
    const diffTime = Math.abs(end - start);
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  };

  const getDateForIndex = () => {
    if (!startDate || !endDate || images.length === 0) return '';
    
    const start = new Date(startDate);
    const totalDays = getDaysBetweenDates(new Date(startDate), new Date(endDate));
    const daysPerImage = totalDays / (images.length - 1);
    const currentDate = new Date(start);
    currentDate.setDate(start.getDate() + Math.round(currentIndex * daysPerImage));
    return currentDate.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  if (images.length === 0) {
    return (
      <div className="status-message">
        Select dates and fetch images to begin
      </div>
    );
  }

  return (
    <div className="slideshow-container">
      <div className="slideshow-image-container">
        {isLoading && <div className="image-placeholder" />}
        
        <img
          src={images[currentIndex]}
          alt={`Satellite view ${currentIndex + 1}`}
          className={`slideshow-image ${imageLoaded ? 'entering' : ''}`}
          onLoad={() => {
            setIsLoading(false);
            setImageLoaded(true);
          }}
        />
        
        <div className="date-indicator">
          <Calendar className="inline-block mr-2" size={16} />
          {getDateForIndex()}
        </div>

        <button
          className="nav-button prev"
          onClick={goToPreviousSlide}
          aria-label="Previous image"
        >
          <ChevronLeft size={24} />
        </button>
        
        <button
          className="nav-button next"
          onClick={goToNextSlide}
          aria-label="Next image"
        >
          <ChevronRight size={24} />
        </button>
      </div>

      <div className="timeline-container">
        {images.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            className={`timeline-dot ${index === currentIndex ? 'active' : ''}`}
            aria-label={`Go to image ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default ImageSlideshow;