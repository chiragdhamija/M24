import cv2
import numpy as np
import os

def extract_cloud_cover(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cloud_cover = np.sum(thresholded) / img.size  # Proportion of cloud cover
    return cloud_cover

def check_increased_cloud_coverage(folder_path, threshold_factor=1.5):
    # Get list of image files sorted by filename (assumed to be ordered by date or index)
    image_files = sorted(os.listdir(folder_path))
    
    # Extract cloud coverage for each image
    cloud_coverages = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        cloud_cover = extract_cloud_cover(img_path)
        cloud_coverages.append(cloud_cover)
    
    if len(cloud_coverages) < 3:
        raise ValueError("Not enough images to perform comparison. At least 3 images are required.")

    # Calculate the average cloud coverage for all images except the last 2
    average_cloud_cover = np.mean(cloud_coverages[:-2])
    
    # Calculate the average cloud coverage for the last 2 images
    last_2_average = np.mean(cloud_coverages[-2:])
    
    # Return 1 if the last 2 images have much greater cloud coverage than the rest, 0 otherwise
    if last_2_average > threshold_factor * average_cloud_cover:
        return "1"
    else:
        return "0"

# Example usage:
folder_path = "images"  # Path to the folder containing the 15 images
def main():
    result = check_increased_cloud_coverage(folder_path, threshold_factor=1.5)
    return result

