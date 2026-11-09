import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import string

# Load trained letter-recognition model (if available)
try:
    model = tf.keras.models.load_model("letters_model.keras")
except Exception:
    model = None

alphabet = string.ascii_uppercase

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open webcam with higher resolution request
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera.")

# Create resizable windows so it doesn't look small (1/4 screen)
cv2.namedWindow("VisionWrite - Air Writing", cv2.WINDOW_NORMAL)
cv2.namedWindow("Writing Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("VisionWrite - Air Writing", 960, 540)
cv2.resizeWindow("Writing Canvas", 640, 480)

canvas = None
prev_point = None
smoothed_point = None
ALPHA = 0.4  # Smoothing factor for EMA (prevents shaky writing)

recognized_text = ""
status = "Show your hand to the camera"
last_action = None
last_action_time = 0
COOLDOWN = 1.0

def classify_gesture(lm):
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    fingers = [index_up, middle_up, ring_up, pinky_up]
    thumb_up = lm[4].y < lm[3].y

    if thumb_up and not any(fingers):
        return "SPACE"
    if all(fingers):
        return "PREDICT"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PAUSE"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "DRAW"
    if not any(fingers) and not thumb_up:
        return "CLEAR"
    return "PAUSE"

def detect_and_correct_shape(canvas_img):
    """Detects geometric shapes and draws a clean corrected version."""
    contours, _ = cv2.findContours(canvas_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return canvas_img, "No drawing found"

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 500:
        return canvas_img, "Drawing too small"

    x, y, w, h = cv2.boundingRect(cnt)
    center = (x + w // 2, y + h // 2)
    radius = max(w, h) // 2

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    vertices = len(approx)

    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    canvas_img[:] = 0  # Clear canvas for clean shape

    if circularity > 0.75:
        cv2.circle(canvas_img, center, radius, 255, 12, cv2.LINE_AA)
        return canvas_img, "Shape Corrected: Circle"
    elif vertices == 3:
        cv2.polylines(canvas_img, [approx], True, 255, 12, cv2.LINE_AA)
        return canvas_img, "Shape Corrected: Triangle"
    elif vertices == 4:
        aspect_ratio = float(w) / h
        if 0.85 <= aspect_ratio <= 1.15:
            cv2.rectangle(canvas_img, (x, y), (x + w, y + h), 255, 12, cv2.LINE_AA)
            return canvas_img, "Shape Corrected: Square"
        else:
            cv2.rectangle(canvas_img, (x, y), (x + w, y + h), 255, 12, cv2.LINE_AA)
            return canvas_img, "Shape Corrected: Rectangle"
    else:
        cv2.drawContours(canvas_img, [approx], -1, 255, 12, cv2.LINE_AA)
        return canvas_img, "Shape Smoothed"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((height, width), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO HAND"
    current_point = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark
        gesture = classify_gesture(lm)

        raw_point = (
            int(lm[8].x * width),
            int(lm[8].y * height)
        )

        # Smooth out hand jitter using Exponential Moving Average
        if smoothed_point is None:
            smoothed_point = raw_point
        else:
            smoothed_point = (
                int(ALPHA * raw_point[0] + (1 - ALPHA) * smoothed_point[0]),
                int(ALPHA * raw_point[1] + (1 - ALPHA) * smoothed_point[1])
            )

        current_point = smoothed_point

        if gesture == "DRAW":
            if prev_point is not None:
                cv2.line(
                    canvas, prev_point, current_point,
                    255, 12, cv2.LINE_AA
                )
            prev_point = current_point
        else:
            prev_point = None
    else:
        prev_point = None
        smoothed_point = None

    now = time.time()

    if gesture in ("DRAW", "PAUSE", "NO HAND"):
        last_action = None

    elif (
        gesture in ("PREDICT", "CLEAR", "SPACE")
        and gesture != last_action
        and now - last_action_time > COOLDOWN
    ):
        if gesture == "PREDICT":
            canvas, status = detect_and_correct_shape(canvas)
            
            # Predict letter if shape correction wasn't triggered
            if model is not None and "Shape Corrected" not in status:
                coords = cv2.findNonZero(canvas)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    crop = canvas[y:y+h, x:x+w]
                    scale = 20.0 / max(w, h)
                    nw = max(1, int(w * scale))
                    nh = max(1, int(h * scale))
                    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

                    image = np.zeros((28, 28), dtype=np.uint8)
                    ox = (28 - nw) // 2
                    oy = (28 - nh) // 2
                    image[oy:oy+nh, ox:ox+nw] = crop

                    model_input = (image.astype("float32") / 255.0)[None, :, :, None]
                    probabilities = model.predict(model_input, verbose=0)[0]
                    index = int(np.argmax(probabilities))
                    confidence = float(probabilities[index])

                    if confidence >= 0.45:
                        letter = alphabet[index]
                        recognized_text += letter
                        status = f"Letter: {letter} ({confidence * 100:.1f}%)"

        elif gesture == "CLEAR":
            canvas[:] = 0
            prev_point = None
            status = "Drawing cleared"

        elif gesture == "SPACE":
            if recognized_text and not recognized_text.endswith(" "):
                recognized_text += " "
            status = "Space added"

        last_action = gesture
        last_action_time = now

    # Overlay drawing onto the webcam feed
    overlay = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    frame = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)

    cv2.putText(frame, f"Gesture: {gesture}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    shown_text = recognized_text[-45:]
    cv2.putText(frame, f"Text: {shown_text}", (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("VisionWrite - Air Writing", frame)
    cv2.imshow("Writing Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()