"""import cv2
import os

# Function to collect cropped grayscale image samples with faces using Haar cascade
def collect_samples_with_faces(output_folder, num_samples):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return

    # Loop to collect samples with faces
    sample_count = 0
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Save the frame if a face is detected
        for (x, y, w, h) in faces:
            # Crop the face region
            face = gray[y:y+h, x:x+w]

            # Save the sample image
            sample_filename = os.path.join(output_folder, f"sample_{sample_count}.jpg")
            cv2.imwrite(sample_filename, face)
            print(f"Sample {sample_count+1}/{num_samples} saved.")
            sample_count += 1

    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
#output_folder = "face_samples_gray"
#num_samples = 100  # Number of samples to collect

#collect_samples_with_faces(output_folder, num_samples)

import cv2
import os

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory for storing the dataset
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize variables
person_id = 1 # Initialize with the ID of the first person
cv2.namedWindow("Capturing Images", cv2.WINDOW_NORMAL)

flag = True  # Flag to control the outer loop

while person_id <= 5 and flag:  # Assuming you want to capture images for 5 people
    # Create a subdirectory for the current person
    person_directory = f"dataset/person_{person_id}"
    if not os.path.exists(person_directory):
        os.makedirs(person_directory)

    count = 0  # Reset the count for each person

    while count < 100:
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop and save only the face region
            face_gray = gray[y:y + h, x:x + w]  # Convert the face region to grayscale
            count += 1
            filename = f"{person_directory}/image_{count}.jpg"
            cv2.imwrite(filename, face_gray)

            # Display the count within the directory
            text = f"Person {person_id}, Image {count}/100"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capturing Images", frame)

            # Wait for a moment before capturing the next image (you can adjust the delay if needed)
            cv2.waitKey(100)  # 100 milliseconds

        if count >= 100:
            flag = False  # Stop capturing after 200 images for the first person
            break

    person_id += 1

cap.release()
cv2.destroyAllWindows()


import cv2
import os

def preprocess_dataset(input_folder, output_folder, target_size=(100, 100), apply_equalization=False):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each subfolder (representing different individuals) in the input folder
    for person_folder in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_folder)
        if os.path.isdir(person_path):
            # Create output subfolder for the current person
            output_person_folder = os.path.join(output_folder, person_folder)
            if not os.path.exists(output_person_folder):
                os.makedirs(output_person_folder)

            # Loop through the images in the current person's folder
            for filename in os.listdir(person_path):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    # Read the image
                    image_path = os.path.join(person_path, filename)
                    image = cv2.imread(image_path)

                    # Convert image to grayscale
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Resize the image to the target size
                    resized_image = cv2.resize(gray_image, target_size)

                    # Optionally apply histogram equalization
                    if apply_equalization:
                        equalized_image = cv2.equalizeHist(resized_image)
                        # Save the equalized image
                        output_path = os.path.join(output_person_folder, filename)
                        cv2.imwrite(output_path, equalized_image)
                    else:
                        # Save the resized image
                        output_path = os.path.join(output_person_folder, filename)
                        cv2.imwrite(output_path, resized_image)

# Example usage:
input_folder = "dataset"  # Input folder containing subfolders for each individual
output_folder = "preprocessed_dataset"  # Output folder for preprocessed images
target_size = (100, 100)  # Target size for resizing the images
apply_equalization = True  # Whether to apply histogram equalization

preprocess_dataset(input_folder, output_folder, target_size, apply_equalization)

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

import cv2
print(dir(cv2))
import os
import numpy as np
# Function to load and preprocess images from subfolders
def load_images_from_subfolders(root_folder):
    images = []
    labels = []
    for person_folder in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            label = int(person_folder.split("_")[1])  # Extract numeric part of folder name as label
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

# Load images and labels from the preprocessed dataset folder
images, labels = load_images_from_subfolders("preprocessed_dataset")

# Create LBPH face recognizer
lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the LBPH model
lbph_face_recognizer.train(images, np.array(labels))

# Save the trained model to a file
lbph_face_recognizer.save("lbph_model.xml")
"""
import cv2
import datetime


# Function to match recognized person's identity with the database and record attendance
import cv2
import datetime
import openpyxl

# Function to match recognized person's identity with the database and record attendance
def record_attendance(recognized_name, department, subject):
    # Check if recognized_name exists in the database
    if recognized_name in attendance_database:
        # Check if the person has already been recognized
        if recognized_name not in recognized_names:
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Record attendance by timestamping the recognized person's presence
            attendance_record = [recognized_name, department, subject, timestamp]

            # Append the attendance record to the attendance log in the Excel sheet
            sheet.append(attendance_record)

            # Add the recognized name to the set of recognized names
            recognized_names.add(recognized_name)

            print(f"Attendance recorded for {recognized_name} ({department}, {subject}) at {timestamp}")
    else:
        print(f"{recognized_name} is not in the database")

# Load the workbook
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = "Attendance Log"
sheet.append(["Name", "Department", "Subject", "Timestamp"])

# Simulated attendance database (replace this with your actual database)
attendance_database = {
    "Person 1": {"department": "CSE", "subject": "Maths"},
    "Person 2": {"department": "CSE", "subject": "Maths"},
    # Add more names with department and subject information as needed
}

# Set to store recognized names to ensure attendance is recorded only once
recognized_names = set()

def recognize_and_track_attendance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform face recognition for each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Perform face recognition
        label, confidence = lbph_face_recognizer.predict(face_roi)

        # Get the name corresponding to the recognized label
        recognized_name = label_name_map.get(label, "Unknown")

        # Get department and subject information from the attendance database
        person_info = attendance_database.get(recognized_name, {})
        department = person_info.get("department", "Unknown Department")
        subject = person_info.get("subject", "Unknown Subject")

        # Display the recognized name and confidence level
        cv2.putText(image, f'Name: {recognized_name}, Department: {department}, Subject: {subject}, Confidence: {confidence}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Record attendance for the recognized person
        record_attendance(recognized_name, department, subject)

    return image

# Load the LBPH model
lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
lbph_face_recognizer.read("lbph_model.xml")

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a mapping from label IDs to names
label_name_map = {
    1: "Person 1",
    2: "Person 2",
    # Add more mappings as needed
}

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face recognition and attendance tracking on the frame
    frame = recognize_and_track_attendance(frame)

    # Display the result
    cv2.imshow('Face Recognition and Attendance Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the workbook
workbook.save("attendance_log.xlsx")

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
