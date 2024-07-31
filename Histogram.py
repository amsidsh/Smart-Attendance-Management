import os
import cv2
from skimage.feature import local_binary_pattern
from skimage import exposure
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Function to extract LBP features
def extract_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)  # Normalize
    return hist_lbp


# Function to extract HOG features
def extract_hog_features(image):
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          visualize=True, block_norm='L2-Hys')
    hog_features = hog_features.reshape(-1)  # Flatten HOG features
    return hog_features


# Path to the directory containing preprocessed image folders
preprocessed_folder = 'preprocessed_dataset'

# Loop through each folder
for person_folder in os.listdir(preprocessed_folder):
    person_path = os.path.join(preprocessed_folder, person_folder)
    if os.path.isdir(person_path):
        print("Processing images in", person_path)

        # Loop through each image in the folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                print("Processing", image_path)

                # Load image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Extract features
                lbp_features = extract_lbp_features(image)
                hog_features = extract_hog_features(image)

                # Print or use the extracted features
                print("LBP Features:", lbp_features)
                print("HOG Features:", hog_features)

              # Here you can save or further process the features as needed
