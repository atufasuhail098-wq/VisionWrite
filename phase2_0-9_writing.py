import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import mediapipe as mp

print("Loading MNIST Dataset (0-9)...")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training CNN Model...")
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=1)
print("AI Model Ready!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0

# Neon Glow Colors (BGR Format)
neon_outer_color = (255, 100, 0)   # Deep Blue Glow
neon_inner_color = (255, 255, 0)   # Bright Cyan/Electric Core
brush_thickness = 10

prediction_text = "Draw with Index Finger & press 'P'"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    drawing_active = False
    current_x, current_y = 0, 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            current_x, current_y = int(index_tip.x * w), int(index_tip.y * h)

            # Drawing condition: Index finger upar aur middle finger niche
            if index_tip.y < index_pip.y and middle_tip.y > middle_pip.y:
                drawing_active = True
                cv2.circle(frame, (current_x, current_y), 8, (0, 255, 0), cv2.FILLED)

    if drawing_active:
        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = current_x, current_y

        # Smooth movement smoothing factor
        smooth_x = int(prev_x + (current_x - prev_x) * 0.4)
        smooth_y = int(prev_y + (current_y - prev_y) * 0.4)

        # --- NEON GLOW DRAWING EFFECT ---
        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), neon_outer_color, brush_thickness + 6)
        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), neon_inner_color, 3)

        prev_x, prev_y = smooth_x, smooth_y
    else:
        # Jab ungli niche ho ya gesture na ho, tab points ko reset karein taaki naya stroke bane
        prev_x, prev_y = 0, 0

    # Blend canvas and frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # UI Banner
    cv2.putText(frame, prediction_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Controls: [P] Predict Digit | [C] Clear Canvas | [Q] Quit", (30, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Phase 1: Air Writing", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        prediction_text = "Canvas cleared. Draw a digit."

    elif key == ord('p'):
        gray_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)

        if coords is not None:
            x_box, y_box, w_box, h_box = cv2.boundingRect(coords)
            cropped = binary[y_box:y_box + h_box, x_box:x_box + w_box]

            scale = 20.0 / max(w_box, h_box)
            new_w = max(1, int(w_box * scale))
            new_h = max(1, int(h_box * scale))

            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

            digit_img = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2

            digit_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            cv2.imshow("What the AI Sees", cv2.resize(digit_img, (280, 280), interpolation=cv2.INTER_NEAREST))

            normalized = digit_img.astype("float32") / 255.0
            prediction_input = np.expand_dims(normalized, axis=0)

            predictions = model.predict(prediction_input, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100

            prediction_text = f"Predicted: {predicted_digit} ({confidence:.1f}% confidence)"
        else:
            prediction_text = "No drawing detected! Draw a digit first."

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()