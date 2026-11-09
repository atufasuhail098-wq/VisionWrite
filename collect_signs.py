import os
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if os.path.exists("sign_data.npy") and os.path.exists("sign_labels.npy"):
    data = list(np.load("sign_data.npy"))
    labels = list(np.load("sign_labels.npy"))
    print(f"Loaded existing data. Total samples so far: {len(data)}")
else:
    data = []
    labels = []


current_label = "A"
print(f"Recording signs for: {current_label}. Press space to capture sample, type letter 'A'-'Z' to change, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    cv2.putText(frame, f"Target Letter: {current_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32 and result.multi_hand_landmarks:
        data.append(landmarks)
        labels.append(ord(current_label) - ord('A'))
        print(f"Captured sample for {current_label} (Total: {len(data)})")
    elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
            current_label = chr(key).upper()
            print(f"Switched target letter to: {current_label}")
cap.release()
cv2.destroyAllWindows()

np.save("sign_data.npy", np.array(data))
np.save("sign_labels.npy", np.array(labels))
print("Saved sign_data.npy and sign_labels.npy successfully!")