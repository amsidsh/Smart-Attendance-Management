import cv2
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
person_id = 4# Initialize with the ID of the first person
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