"""
Description : Captures hand gesture landmarks using Mediapipe and saves them with corresponding labels.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Arrays to store gesture data and labels
data = []
labels = []

# Instructions for user
print("Press 's' to save a frame for the current gesture.")
print("Press 'q' to quit.")

# Main loop for data collection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmarks (x, y, z)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # Show current frame
            cv2.imshow('ASL Data Collection', frame)

            # Save data if 's' is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Gesture saved. Enter the label for this gesture (e.g., 'A'):")
                label = input()
                data.append(landmarks)
                labels.append(label)
                print(f"Saved gesture with label: {label}")

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data
np.save('data.npy', np.array(data))
np.save('labels.npy', np.array(labels))
print("Data and labels saved successfully!")
