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
