from flask import Flask, jsonify, request
from flask_cors import CORS
import ee
import os
import requests
from google.oauth2 import service_account

# Set the path to your service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./ee-chiragdhamija0203-2a464e556b15.json"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Function to initialize Google Earth Engine
def initialize_earth_engine():
    try:
        # Create credentials from the service account JSON key file with scopes
        scopes = ['https://www.googleapis.com/auth/earthengine.readonly']
        credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=scopes
        )
        # Initialize Earth Engine with the credentials
        ee.Initialize(credentials)
        print("Google Earth Engine initialized successfully!")
    except Exception as e:
        print("Error initializing Google Earth Engine:", e)
        raise e  # Raise the exception to stop further execution

# Call the initialization function
initialize_earth_engine()

@app.route('/google-earth-api', methods=['GET'])
def google_earth_api():
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')

    # Parse dates from the request
    try:
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date)
    except Exception as e:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400

    # Create an ImageCollection from MODIS
    dataset = ee.ImageCollection('MODIS/061/MOD09GA') \
              .filter(ee.Filter.date(start_date, end_date))

    # Define the region of interest as a rectangle
    region = ee.Geometry.Rectangle([60, 5, 100, 35])  # Corrected to use (minLon, minLat, maxLon, maxLat)

    # Select the true color bands
    trueColor = dataset.select(['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'])

    # Check if the dataset has images
    image_count = trueColor.size().getInfo()
    if image_count == 0:
        return jsonify({'error': 'No images available for the specified date range.'}), 404

    save_dir = 'images'
    os.makedirs(save_dir, exist_ok=True)
    # Generate image URLs for each day in the range
    image_urls = []
    date_range = ee.DateRange(start_date, end_date)

    # Convert the date range to a list of timestamps
    date_list = ee.List.sequence(date_range.start().millis(), date_range.end().millis(), 24 * 60 * 60 * 1000).getInfo()

    # Loop through each date in the range
    for date in date_list:
        image = trueColor.filterDate(ee.Date(date)).mean().clip(region)  # Get the mean image for the day and clip it to the region
        if image.bandNames().size().getInfo() == 0:  # Check if the image has any bands
            continue  # Skip if there are no bands

        url = image.getThumbUrl({'min': -100.0, 'max': 8000.0, 'dimensions': '512x512'})
        image_urls.append(url)
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{save_dir}/image_{ee.Date(date).format('YYYYMMdd').getInfo()}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)

    return jsonify({'images': image_urls})

if __name__ == '__main__':
    app.run(debug=True)
