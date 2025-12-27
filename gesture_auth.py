import cv2
import time

cap = cv2.VideoCapture(0)

gesture_verified = False
start_time = time.time()

print("Show OPEN PALM clearly to verify gesture")

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
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        # VERY IMPORTANT: Higher threshold to avoid noise
        if area > 25000:
            # Wait at least 2 seconds before accepting
            if time.time() - start_time > 2:
                gesture_verified = True
                cv2.putText(frame, "GESTURE VERIFIED",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Authentication", frame)

    # EXIT ONLY AFTER GESTURE OR 'q'
    if gesture_verified:
        cv2.waitKey(1000)  # show success for 1 sec
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if gesture_verified:
    print("Gesture authentication successful!")
else:
    print("Gesture authentication failed!")
