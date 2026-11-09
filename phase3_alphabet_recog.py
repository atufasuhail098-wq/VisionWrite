import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load your newly trained custom model
model = tf.keras.models.load_model("sign_model.keras")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting live recognition... Press 'q' or ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_char = ""

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Extract 21 landmarks (x, y, z) -> total 63 features
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Reshape for model input
            input_data = np.array([landmarks])

           # Predict the letter
            prediction = model.predict(input_data, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            # Convert class index (0-25) to letter (A-Z)
            predicted_char = chr(class_idx + ord('A'))

            # Terminal par print karwayenge taaki pata chale model kya read kar raha hai
            print(f"Predicted: {predicted_char} with confidence: {confidence*100:.2f}%")

            # Screen par direct dikhayein bina confidence roke
            cv2.putText(frame, f"Sign: {predicted_char} ({confidence*100:.1f}%)", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.namedWindow("Custom Sign Language Recognition", cv2.WINDOW_NORMAL)
    cv2.imshow("Custom Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

