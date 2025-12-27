import os

if not os.path.exists("dataset/user1") or len(os.listdir("dataset/user1")) == 0:
    print("No face dataset found.")
    print("Please run face_dataset.py to register first.")
    exit()


import cv2
import os
import numpy as np
import time

# ================= FACE AUTHENTICATION ================= #

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

dataset_path = "dataset/user1"
faces = []
labels = []

for img_name in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, img_name), cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    labels.append(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

cap = cv2.VideoCapture(0)

face_verified = False
face_start_time = None

print("STEP 1: FACE AUTHENTICATION")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cv2.putText(frame, "Hold your face steady",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    if len(detected_faces) == 0:
        face_start_time = None

    for (x, y, w, h) in detected_faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        if confidence < 70:
            if face_start_time is None:
                face_start_time = time.time()

            elapsed = time.time() - face_start_time

            cv2.putText(frame, f"Verifying... {int(elapsed)}s",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)

            if elapsed >= 2:
                face_verified = True
                cv2.putText(frame, "FACE VERIFIED",
                            (x, y-40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
        else:
            face_start_time = None

    cv2.imshow("Face Authentication", frame)

    if face_verified:
        cv2.waitKey(1200)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if not face_verified:
    print("ACCESS DENIED ❌ (Face Failed)")
    exit()

print("Face authentication successful ✅")

# ================= GESTURE AUTHENTICATION ================= #

cap = cv2.VideoCapture(0)

gesture_verified = False
gesture_start_time = None

print("STEP 2: GESTURE AUTHENTICATION (OPEN PALM)")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.putText(frame, "Show OPEN PALM",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 25000:
            if gesture_start_time is None:
                gesture_start_time = time.time()

            elapsed = time.time() - gesture_start_time

            cv2.putText(frame, f"Verifying... {int(elapsed)}s",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)

            if elapsed >= 2:
                gesture_verified = True
                cv2.putText(frame, "GESTURE VERIFIED",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
        else:
            gesture_start_time = None

    else:
        gesture_start_time = None

    cv2.imshow("Gesture Authentication", frame)

    if gesture_verified:
        cv2.waitKey(1200)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if gesture_verified:
    print("ACCESS GRANTED 🎉🎉")
else:
    print("ACCESS DENIED ❌ (Gesture Failed)")
