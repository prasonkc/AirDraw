import cv2
import mediapipe as mp

# Initialize hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Initialize face module
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.8)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read() # Ret = return value that detects whether frame was successfully read or not

    # Skip if frame wasnt successfully read
    if not ret:
        continue

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and face
    hand_results = hands.process(frame_rgb)
    face_results = face_detection.process(frame_rgb)

    # --------------------HANDS----------------------------------------
    # Check if hands are detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --------------------FACE--------------------------------------------
    # Check if face are detected
    if face_results.detections:
        for face in face_results.detections:
            mp_drawing.draw_detection(frame, face)

    # Display the frame with hand + face landmarks
    cv2.imshow('Hand + Face Recognizations', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()