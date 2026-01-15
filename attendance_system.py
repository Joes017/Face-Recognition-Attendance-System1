import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# 1️⃣ Haar Cascade for face detection
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# 2️⃣ Prepare dataset and train recognizer
# -------------------------
def prepare_training_data(dataset_path="dataset"):
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for student_name in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_path):
            continue

        label_map[label_id] = student_name

        for image_name in os.listdir(student_path):
            image_path = os.path.join(student_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    return faces, np.array(labels), label_map

faces, labels, label_map = prepare_training_data()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# -------------------------
# 3️⃣ CSV Attendance file
# -------------------------
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

# -------------------------
# 4️⃣ Function to mark attendance
# -------------------------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Read existing CSV
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if student already marked today
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_row = pd.DataFrame({"Name": [name], "Date": [date], "Time": [time]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for {name}")

# -------------------------
# 5️⃣ Start camera for recognition
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        # Recognize face
        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = label_map[label]
            mark_attendance(name)
        else:
            name = "Unknown"

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
