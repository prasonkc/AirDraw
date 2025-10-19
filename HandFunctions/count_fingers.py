def count_fingers(hand_landmarks, hand_label):
    """
    hand_landmarks: hand's landmarks
    hand_label: 'Left' or 'Right'
    """

    # Define Finger tips and finger pips (pips = middle joints)
    finger_tips = [4,8,12,16,20]
    finger_pips = [2,6,10,14,18]

    # Only track the index finger
    finger_tip = 8

    fingers=[]

# Logic1: When opening a fist, thumbs move sideways so we compare x coordinates. Other fingers move up and down so we compare y coordinates
# Logic2: If the tip is above the middle joint, the finger is considered raised.

# Check for thumb

    # Thumbs move in opposite x direction in each hand so we check whether its right hand or left hand 
    # if hand_label == "Right":
    #     fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x else 0) # Thumb landmarks = 2 for pip and 4 for tip
    # else:
    #     fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x else 0)


# Check for other fingers
    # for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
    #     fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)


    # Return sum of the fingers to find out how many fingers are raised
    return sum(fingers)