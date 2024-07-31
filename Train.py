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
