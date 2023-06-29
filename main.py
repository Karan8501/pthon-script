import cv2
import numpy as np
import os
import pandas as pd
import time

# Load the face cascade XML file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize some variables
known_face_encodings = []
known_face_roll_no = []
face_names = []
attendance_record = set([])
roll_record = {}

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Rows in log file
name_col = []
roll_no_col = []
time_col = []

df = pd.read_excel("students" + os.sep + "students_db.xlsx")

for key, row in df.iterrows():
    roll_no = row["roll_no"]
    name = row["name"]
    image_path = row["image"]
    roll_record[roll_no] = name
    try:
        student_image = cv2.imread("../public/assets/uploads" + os.sep + image_path)
        student_image_gray = cv2.cvtColor(student_image, cv2.COLOR_BGR2GRAY)
        face_encoding = face_recognition.face_encodings(student_image_gray)[0]
        known_face_encodings.append(face_encoding)
        known_face_roll_no.append(roll_no)
    except:
        print("../public/assets/uploads" + os.sep + image_path + " Student has not uploaded an image")
        continue

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the current frame from video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_image = frame[y:y+h, x:x+w]

        # Resize the face image to a fixed size for face recognition
        face_image_resized = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)

        # Convert the resized face image to grayscale
        face_image_gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)

        # Perform face recognition on the face image
        face_encodings = face_recognition.face_encodings(face_image_gray)

        if len(face_encodings) > 0:
            # Compare the face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.5)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                roll_no = known_face_roll_no[match_index]
                name = roll_record[roll_no]

                if roll_no not in attendance_record:
                    attendance_record.add(roll_no)
                    print(name, roll_no)
                    name_col.append(name)
                    roll_no_col.append(roll_no)
                    curr_time = time.localtime()
                    curr_clock = time.strftime("%H:%M:%S", curr_time)
                    time_col.append(curr_clock)

            face_names.append(name)

    # Display the face rectangles and names on the frame
    for (x, y, w, h), name in zip(faces, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Calculate and display the FPS (Frames Per Second)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
