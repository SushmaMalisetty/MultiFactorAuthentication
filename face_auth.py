import cv2
import os
import numpy as np
import time

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load dataset
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
start_time = time.time()

print("Look at the camera for FACE authentication")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cv2.putText(frame, "Hold your face steady",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    for (x, y, w, h) in detected_faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face)

        # Draw rectangle always
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Stable condition
        if confidence < 70 and time.time() - start_time > 2:
            face_verified = True
            cv2.putText(frame, "FACE VERIFIED",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

    cv2.imshow("Face Authentication", frame)

    # Exit only after verification
    if face_verified:
        cv2.waitKey(1000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if face_verified:
    print("Face authentication successful!")
else:
    print("Face authentication failed!")
