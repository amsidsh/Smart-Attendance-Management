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