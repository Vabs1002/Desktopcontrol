import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyautogui
import os
import time

# --- PRODUCT CONFIGURATION ---
# Disable pyautogui delay for faster mouse, but we will add custom sleeps for stability
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True 

# [cite_start]1. Load the Model and Label Encoder [cite: 1, 49]
if not os.path.exists('models/gesture_model.h5') or not os.path.exists('models/label_encoder.pkl'):
    print("Error: Model files not found. Run training.py first!")
    exit()

model = tf.keras.models.load_model('models/gesture_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 2. Setup MediaPipe & Screen Size
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.8, # Higher confidence for "Nice Product" feel
    min_tracking_confidence=0.8
)

screen_w, screen_h = pyautogui.size()

# 3. Mouse Smoothing Variables
# We use a moving average filter to stop cursor jitter
prev_x, prev_y = 0, 0
smooth_factor = 7 

cap = cv2.VideoCapture(0)
print("--- GesturePro: Universal Product Controller Active ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    f_h, f_w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # [cite_start]Predict Gesture [cite: 1, 49]
            prediction = model.predict(np.array([landmarks]), verbose=0)
            class_id = np.argmax(prediction)
            gesture_name = label_encoder.inverse_transform([class_id])[0]
            confidence = np.max(prediction)

            # --- PRODUCT LOGIC ENGINE ---
            if confidence > 0.90: # Only trigger if AI is 90% sure
                cv2.putText(frame, f"Active: {gesture_name}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # FEATURE 1: PRECISION MOUSE (Point)
                if gesture_name == "Point":
                    index_tip = hand_landmarks.landmark[8]
                    # Map coordinates with an 'inner box' for easier reach
                    raw_x = np.interp(index_tip.x, [0.2, 0.8], [0, screen_w])
                    raw_y = np.interp(index_tip.y, [0.2, 0.8], [0, screen_h])
                    
                    # Apply Smoothing Logic
                    curr_x = prev_x + (raw_x - prev_x) / smooth_factor
                    curr_y = prev_y + (raw_y - prev_y) / smooth_factor
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                # FEATURE 2: CLICK (Pinch)
                elif gesture_name == "Pinch":
                    pyautogui.click()
                    time.sleep(0.3) # Prevent double-click spam

                # FEATURE 3: PRIVACY GUARD (Fist) - The 'Boss Key'
                elif gesture_name == "Fist":
                    print("✊ Privacy Guard Triggered!")
                    pyautogui.hotkey('win', 'd')      # Minimize all windows
                    pyautogui.press('volumemute')     # Mute sound
                    time.sleep(1.0)                   # Cooldown to prevent loop

                # FEATURE 4: MEDIA CONTROL (Palm)
                elif gesture_name == "Palm_Open":
                    pyautogui.press('playpause')
                    time.sleep(0.5)

                # FEATURE 5: SYSTEM VOLUME (Thumbs)
                elif gesture_name == "Thumb_Up":
                    pyautogui.press('volumeup')
                elif gesture_name == "Thumb_Down":
                    pyautogui.press('volumedown')

                # FEATURE 6: NAVIGATION (Peace)
                elif gesture_name == "Peace":
                    pyautogui.hotkey('ctrl', 'tab')   # Next Browser Tab
                    time.sleep(0.4)
                
                # FEATURE 7: VIRTUAL DESKTOPS (Swipes)
                elif gesture_name == "Swipe_R":
                    pyautogui.hotkey('ctrl', 'win', 'right')
                    time.sleep(0.7)
                elif gesture_name == "Swipe_L":
                    pyautogui.hotkey('ctrl', 'win', 'left')
                    time.sleep(0.7)

    cv2.imshow('GesturePro v1.0 - Product Interface', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()