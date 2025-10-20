import cv2
import mediapipe as mp
import numpy as np
import time

# Functions import
from HandFunctions.count_fingers import count_fingers

# Initialize hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)

# Initialize face module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, max_num_faces=1)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cam = cv2.VideoCapture(0)

# Create a canvas for drawing
canvas = None
prev_point = None
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
brush_size = 5
color_flag = 0
brush_color = colors[color_flag]
last_color_change = 0


# Track the state of drawing 
allow_draw = True

while cam.isOpened():
    ret, frame = cam.read() # Ret = return value that detects whether frame was successfully read or not

    # Skip if frame wasnt successfully read
    if not ret:
        continue

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the canvas, the same size of frame
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Process the frame to detect hands and face
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    # --------------------HANDS----------------------------------------
    # Check if hands are detected
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        hand_handedness = hand_results.multi_handedness[0]
        num_fingers_up = count_fingers(hand_landmarks, hand_handedness)

        h, w, _ = frame.shape
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        prev_x, prev_y = 0, 0
        smooth_factor = 2

        # Smoothing the coordinates
        smooth_x = int(prev_x + (x - prev_x) / smooth_factor)
        smooth_y = int(prev_y + (y - prev_y) / smooth_factor)

        # Fist = pause drawing
        allow_draw = num_fingers_up != 0

        # Erase = 4 fingers
        if num_fingers_up == 4:
            canvas = np.zeros_like(frame)

        # Color cycling
        current_time = time.time()
        if num_fingers_up == 3 and current_time - last_color_change > 0.5:
            color_flag = (color_flag + 1) % len(colors)
            brush_color = colors[color_flag]
            last_color_change = current_time
        elif num_fingers_up == 2 and current_time - last_color_change > 0.5:
            color_flag = (color_flag - 1) % len(colors)
            brush_color = colors[color_flag]
            last_color_change = current_time

        # Drawing
        if prev_point is not None and allow_draw:
            cv2.line(canvas, prev_point, (smooth_x, smooth_y), brush_color, brush_size)
        
        
        prev_point = (smooth_x, smooth_y)
        prev_x, prev_y = smooth_x, smooth_y

    # --------------------FACE--------------------------------------------
    # Check if face are detected
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #         frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
    #         mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

    # Overlay canvas on frame
    frame = cv2.add(frame, canvas)

    # Display the frame with hand + face landmarks
    cv2.imshow('Hand + Face Recognizations', canvas)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()