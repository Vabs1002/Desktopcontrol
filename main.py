
```python
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyautogui

# Load the brain and the label mapping
model = tf.keras.models.load_model('models/gesture_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

print("GesturePro is active. Control your desktop now!")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    [cite_start]frame = cv2.flip(frame, 1) # Mirror view for UX [cite: 55]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmarks
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])
            
            # Predict
            prediction = model.predict(np.array([lm_list]), verbose=0)
            classID = np.argmax(prediction)
            gesture = encoder.inverse_transform([classID])[0]
            confidence = np.max(prediction)

            # [cite_start]High confidence threshold for error prevention [cite: 52]
            if confidence > 0.90:
                cv2.putText(frame, f"{gesture} ({int(confidence*100)}%)", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # [cite_start]Action Mapping [cite: 15, 43, 46]
                if gesture == "Palm_Open": pyautogui.press('playpause')
                elif gesture == "Thumb_Up": pyautogui.press('volumeup')
                elif gesture == "Thumb_Down": pyautogui.press('volumedown')
                elif gesture == "Peace": pyautogui.hotkey('ctrl', 'tab')
                elif gesture == "Fist": pyautogui.press('volumemute')
                # Add more mappings as you train more gestures!

    cv2.imshow('GesturePro Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
