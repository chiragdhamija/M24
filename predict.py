import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.model_selection import train_test_split


time_series_data = []
num_groups = 1
num_images_per_group = 15
num_clusters = 8  # Number of clusters from K-means


def kmeans_segmentation(image_path, k=8):
    # Add a random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Step 2: Preprocess the image
    pixels = image.reshape((-1, 3))  # Reshape to a 2D array of pixels
    pixels = np.float32(pixels)  # Convert to float32

    # Step 3: Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 (pixel values)
    centers = np.uint8(centers)

    # Step 4: Reconstruct the segmented image
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)  # Reshape to original image shape

    # Step 5: Map each (x, y) coordinate to its corresponding cluster
    clusters = labels.reshape(height, width)

    return clusters  # Return the cluster labels

def extract_cloud_cover(img_path, cluster_label, clusters):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Create a binary mask for the specified cluster
    mask = (clusters == cluster_label).astype(np.uint8) * 255  # Binary mask

    # Check dimensions
    if mask.shape != img.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {img.shape}.")

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Threshold the masked image to get cloud cover
    _, thresholded = cv2.threshold(masked_img, 150, 255, cv2.THRESH_BINARY)
    cloud_cover = np.sum(thresholded) / masked_img.size  # Proportion of cloud cover
    return cloud_cover

def process_images_in_directory(directory, k=8):
    time_series_data = []
    
    for group in range(1, num_groups + 1):
        group_folder = f"group_{group}/"
        
        for img_name in sorted(os.listdir(group_folder))[:num_images_per_group]:
            img_path = os.path.join(group_folder, img_name)
            
            # Check if the file exists
            if not os.path.isfile(img_path):
                print(f"File not found: {img_path}")
                continue

            # Get clusters from K-means segmentation
            clusters = kmeans_segmentation(img_path, k=num_clusters)

            # Extract timestamp from filename
            try:
                date_parts = img_name.split('-')
                if len(date_parts) == 3 and date_parts[2].endswith('.png'):
                    date_string = f"20{date_parts[2][:-4]}-{date_parts[1]}-{date_parts[0]}"  # YYYY-MM-DD
                    
                    # Calculate cloud cover for each cluster
                    for cluster_label in range(num_clusters):
                        cloud_cover = extract_cloud_cover(img_path, cluster_label, clusters)
                        time_series_data.append({
                            'timestamp': date_string, 
                            'group': group, 
                            'cluster': cluster_label, 
                            'cloud_cover': cloud_cover
                        })
                else:
                    print(f"Skipping file due to unexpected format: {img_name}")
            except Exception as e:
                print(f"Error processing file {img_name}: {e}")

    # Create DataFrame
    features_df = pd.DataFrame(time_series_data)
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], errors='coerce')
    features_df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid dates
    return features_df

for group in range(1, num_groups + 1):
    directory = f"group_{group}/"
    print(f"Processing images in {directory}...")
    features_df = process_images_in_directory(directory, k=num_clusters)
    print(features_df)




# Assuming features_df has columns ['timestamp', 'group', 'cluster', 'cloud_cover']
# Normalize 'cloud_cover' values for each cluster
scaler = MinMaxScaler()
features_df['cloud_cover'] = scaler.fit_transform(features_df[['cloud_cover']])

# Organize data by group and cluster, creating sequences
time_series_data = {}
for (group, cluster), group_df in features_df.groupby(['group', 'cluster']):
    group_df = group_df.sort_values('timestamp')
    cloud_cover_sequence = group_df['cloud_cover'].values
    if len(cloud_cover_sequence) >= 4:  # Ensuring minimum sequence length
        time_series_data[(group, cluster)] = cloud_cover_sequence

# Define LSTM model structure
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), MeanAbsoluteError()])
    return model

# Train model for each group and cluster
results = {}
for (group, cluster), sequence in time_series_data.items():
    # Create dataset with past sequences (e.g., look-back = 3)
    look_back = 3
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:i + look_back])
        y.append(sequence[i + look_back])
    X, y = np.array(X), np.array(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape data for LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Model training with EarlyStopping
    model = create_lstm_model((look_back, 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test),
                        callbacks=[early_stopping], verbose=0)
    
    # Evaluate model
    loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Group {group}, Cluster {cluster} - Loss: {loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    results[(group, cluster)] = {'model': model, 'loss': loss, 'mse': mse, 'mae': mae, 'history': history}
    

def predict_cyclone(x, y, image_path, features_df, models, num_runs=5, threshold=0.6):
    # Store predictions from multiple runs
    cyclone_predictions = []

    for _ in range(num_runs):
        # Get the cluster from the K-means segmentation
        clusters = kmeans_segmentation(image_path, k=num_clusters)

        # Check if the coordinates are within bounds
        height, width = clusters.shape
        if not (0 <= x < width and 0 <= y < height):
            print("Coordinates are out of bounds.")
            return

        # Get the cluster label for the provided (x, y) coordinates
        cluster_label = clusters[y, x]

        # Filter features_df for the specific cluster
        cluster_data = features_df[features_df['cluster'] == cluster_label]

        # Check if there is data for the cluster
        if cluster_data.empty:
            print(f"No data found for cluster {cluster_label}.")
            cyclone_predictions.append(False)  # Assume no cyclone likely if no data
            continue

        # Calculate average cloud cover for the cluster
        avg_cloud_cover = cluster_data['cloud_cover'].mean()

        # Predict future cloud cover values using the LSTM model
        if (group, cluster_label) not in models:
            print(f"No model found for group {group} and cluster {cluster_label}.")
            cyclone_predictions.append(False)  # No model means no cyclone likely
            continue
        model = models[(group, cluster_label)]['model']  # Get the trained model for this cluster
        look_back = 3  # Use the same look-back period as used in training

        # Prepare the input sequence for the model
        cloud_cover_sequence = cluster_data['cloud_cover'].values[-look_back:]  # Last 'look_back' values
        input_seq = np.array(cloud_cover_sequence).reshape((1, look_back, 1))  # Reshape for LSTM
        
        # Predict future cloud cover values
        future_forecast = []
        for _ in range(5):  # Predict 5 future time steps
            pred = model.predict(input_seq)
            future_forecast.append(pred[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)  # Update input sequence

        # Check if any of the future predicted values exceed the average cloud cover
        if any(value > avg_cloud_cover for value in future_forecast):
            cyclone_predictions.append(True)
        else:
            cyclone_predictions.append(False)

    # Aggregate results from all runs
    cyclone_likelihood = sum(cyclone_predictions) / num_runs  # Calculate the proportion of "likely" predictions

    # Make final prediction based on threshold
    if cyclone_likelihood >= threshold:
        return 1
    else:
        return 0

    # Output details
    # print(f"Average cloud cover for cluster {cluster_label}: {avg_cloud_cover}")
    # print(f"Predicted future cloud cover: {future_forecast}")
    # print(f"Cyclone likelihood across runs: {cyclone_likelihood:.2%} ({sum(cyclone_predictions)} out of {num_runs} runs)")

# Example usage
x_coord = 120  # Example x coordinate
y_coord = 180  # Example y coordinate
image_path = "images/1-5-19.png"  # Path to the image for K-means segmentation

def main():
    predict_cyclone(x_coord, y_coord, image_path, features_df, results)
