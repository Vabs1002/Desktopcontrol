
import cv2
import mediapipe as mp
import csv
import time
import os

# 1. Define your 10 gestures
gestures = ["Palm_Open", "Fist", "Thumb_Up", "Thumb_Down", "Point", "Pinch", "Peace", "Three_Fingers", "Swipe_L", "Swipe_R"]
[cite_start]samples_per_gesture = 400 # High sample count for reliability [cite: 61]

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# [cite_start]Ensure data folder exists [cite: 61]
if not os.path.exists('data'): os.makedirs('data')

print("--- GesturePro: Data Collection Started ---")

with open('data/gesture_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for gesture in gestures:
        # [cite_start]UX: 3-second prep time ensures clarity [cite: 9]
        print(f"\nPREPARE: {gesture}. Starting in 3s...")
        time.sleep(3) 
        
        count = 0
        while count < samples_per_gesture:
            ret, frame = cap.read()
            [cite_start]frame = cv2.flip(frame, 1) # Mirror for better UX [cite: 51]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        # Extract 21 points (x, y, z)
                        landmarks.extend([lm.x, lm.y, lm.z])
                    writer.writerow(landmarks + [gesture])
                    count += 1
            
            # [cite_start]UX: Visual feedback of system state [cite: 56]
            cv2.putText(frame, f"Recording: {gesture} ({count}/{samples_per_gesture})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('GesturePro Collector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
cap.release()
cv2.destroyAllWindows()

```
