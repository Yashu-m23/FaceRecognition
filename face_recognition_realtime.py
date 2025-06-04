import face_recognition
import cv2
import os
import numpy as np

# Load known face encodings
known_face_encodings = []
known_face_names = []

script_dir = os.path.dirname(os.path.abspath(__file__))
known_faces_dir = os.path.join(script_dir, "known_faces")

print("Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Remove extension

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Starting real-time face recognition. Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)

    # Show the frame
    cv2.imshow('Face Recognition - Press Q to Quit', frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
