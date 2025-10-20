# ✋ Air Draw — Hand Gesture Drawing App (OpenCV + Mediapipe)

Draw in the air using your hand gestures — no mouse or touchscreen needed!
This project uses OpenCV and MediaPipe Hands for real-time hand tracking and virtual drawing.

Make sure you have Python 3.8+ installed.
Then install dependencies:
`pip install opencv-python mediapipe numpy`

# Controls
| Gesture                   | Action                   |
| ------------------------- | ------------------------ |
| ✊ (Fist / 1 finger)      | Pause drawing            |
| ✋ (All 5 fingers)        | Clear the canvas         |
| 🤘 (3 fingers up)         | Cycle **previous** color |
| 🖖 (4 fingers up)         | Cycle **next** color     |
| ☝️ (Index only)           | Draw mode                |
| 👉 Move your index finger | Draw on screen           |
