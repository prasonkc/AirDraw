import cv2
import mediapipe as mp
import numpy as np
import time

# Functions import
from HandFunctions.count_fingers import count_fingers

# Initialize hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cam = cv2.VideoCapture(0)

# Create a canvas for drawing
canvas = None
prev_point = None

# Persist previous raw coordinates for smoothing across frames
prev_x = None
prev_y = None

colors = [
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
    (236, 240, 241),
    (44, 62, 80)
]
brushes = [5, 10, 15, 20, 25]
brush_flag = 0
brush_size = brushes[brush_flag]
color_flag = 0
brush_color = colors[color_flag]
last_color_change = 0

# Track the state of drawing
allow_draw = True

# smoothing factor (higher -> smoother but more lag)
SMOOTH_FACTOR = 4

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        hand_handedness = hand_results.multi_handedness[0]
        num_fingers_up = count_fingers(hand_landmarks, hand_handedness)
        label = hand_handedness.classification[0].label

        h, w, _ = frame.shape
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        if label == "Right":
            # Initialize prev_x/prev_y on the first observation
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y
                prev_point = (x, y)

            # Compute smoothed coordinates using previous raw coordinates
            smooth_x = int(prev_x + (x - prev_x) / SMOOTH_FACTOR)
            smooth_y = int(prev_y + (y - prev_y) / SMOOTH_FACTOR)

            # Update drawing state: (original had `num_fingers_up != 1`)
            # keep your original intent but consider: fist=0 to pause drawing.
            allow_draw = (num_fingers_up != 1)

            # Erase = 5 fingers (you used 5 before; keep if intended)
            if num_fingers_up == 5:
                canvas = np.zeros_like(frame)
                prev_point = None  # reset stroke
                prev_x, prev_y = None, None
                # skip drawing this frame
                prev_x, prev_y = None, None
                continue

            # Color cycling with cooldown
            current_time = time.time()
            if num_fingers_up == 4 and current_time - last_color_change > 0.5:
                color_flag = (color_flag + 1) % len(colors)
                brush_color = colors[color_flag]
                last_color_change = current_time
            elif num_fingers_up == 3 and current_time - last_color_change > 0.5:
                color_flag = (color_flag - 1) % len(colors)
                brush_color = colors[color_flag]
                last_color_change = current_time

            # If starting a new stroke, set prev_point
            if prev_point is None:
                prev_point = (smooth_x, smooth_y)

            # Draw line if allowed
            if allow_draw and prev_point is not None:
                # ensure brush size is integer
                cv2.line(canvas, prev_point, (smooth_x, smooth_y), brush_color, int(brush_size))
                prev_point = (smooth_x, smooth_y)
            else:
                # if not drawing, reset prev_point so next stroke starts cleanly
                prev_point = None

            # Save raw coordinates for smoothing next frame
            prev_x, prev_y = x, y

    else:
        # No hand detected: reset prev_x/prev_y so smoothing doesn't jump later
        prev_x, prev_y = None, None
        prev_point = None

    # Blend canvas over frame (add or weighted blend)
    frame = cv2.add(frame, canvas)

    cv2.imshow('Hand + Face Recognizations', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
