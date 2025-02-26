import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# ----------------------------
# Function to load known faces
# ----------------------------
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # List of people and their image paths
    faces = [
        ("Ritesh Singh", "faces/ritesh.jpg"),
        ("Shashwat Rao", "faces/shashwat.jpg"),
        ("V. Venketsh", "faces/venky.jpg")
    ]

    print("Loading known faces...")
    for name, file in faces:
        try:
            image = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✔ Loaded face for: {name}")
            else:
                print(f"❌ No face found in {file}")
        except Exception as e:
            print(f"⚠ Error loading {file}: {e}")

    print("Known faces loaded successfully!\n")
    return known_face_encodings, known_face_names

# ----------------------------
# Function to initialize video capture
# ----------------------------
def initialize_camera():
    print("Initializing video capture...")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("❌ Failed to open camera!")
        exit(1)
    print("✔ Camera initialized successfully!\n")
    return video_capture

# ----------------------------
# Function to create CSV file for attendance
# ----------------------------
def create_csv_file():
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"{current_date}.csv"
    print(f"Creating/Opening attendance file: {csv_filename}\n")

    # Open the CSV file in append mode
    with open(csv_filename, "a+", newline="") as f:
        writer = csv.writer(f)
        # Write header if the file is empty
        if f.tell() == 0:
            writer.writerow(["Name", "Date", "Time"])
    return csv_filename

# ----------------------------
# Function to mark attendance
# ----------------------------
def mark_attendance(name, csv_filename):
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, current_date, current_time])
    print(f"✅ Attendance marked for {name} at {current_time}\n")

# ----------------------------
# Main function to run face recognition
# ----------------------------
def run_attendance_system():
    known_face_encodings, known_face_names = load_known_faces()
    video_capture = initialize_camera()
    csv_filename = create_csv_file()
    students = known_face_names.copy()

    print("🎥 Starting real-time face recognition. Press 'q' to exit.\n")

    while True:
        # Capture frame from the camera
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Failed to capture video frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Process each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            threshold = 0.4

            # Determine if the face matches a known person
            if face_distances[best_match_index] < threshold:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            # Mark attendance if recognized
            if name in students:
                students.remove(name)
                mark_attendance(name, csv_filename)

            # Display bounding box and name
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (0, 255, 0), 2)
            cv2.putText(frame, name, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the video frame
        cv2.imshow("Attendance System", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Exiting attendance system...")
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    print("✔ Resources released. Goodbye!\n")

# ----------------------------
# Run the attendance system
# ----------------------------
if __name__ == "__main__":
    run_attendance_system()
