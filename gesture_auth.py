import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_verified = False
print("Show the SECRET gesture (Index Finger UP)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Index finger tip and middle finger tip
            index_tip = landmarks[8].y
            middle_tip = landmarks[12].y

            # Gesture condition
            if index_tip < middle_tip:
                gesture_verified = True
                cv2.putText(frame, "Gesture Verified",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Show Index Finger Up",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Authentication", frame)

    if gesture_verified or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if gesture_verified:
    print("Gesture authentication successful!")
else:
    print("Gesture authentication failed!")
