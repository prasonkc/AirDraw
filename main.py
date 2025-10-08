import cv2
import mediapipe as mp

# Functions import
from HandFunctions.count_fingers import count_fingers

# Initialize hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=2)

# Initialize face module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, max_num_faces=1)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read() # Ret = return value that detects whether frame was successfully read or not

    # Skip if frame wasnt successfully read
    if not ret:
        continue

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and face
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    # --------------------HANDS----------------------------------------
    # Check if hands are detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Either left or right
            hand_label = hand_handedness.classification[0].label
            num_fingers_up = count_fingers(hand_landmarks, hand_handedness)
            print(f"{hand_label} hand: {num_fingers_up} fingers up")

    # --------------------FACE--------------------------------------------
    # Check if face are detected
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

    # Display the frame with hand + face landmarks
    cv2.imshow('Hand + Face Recognizations', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()