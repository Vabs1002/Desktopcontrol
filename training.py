```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def retrain_model():
    # Load the captured landmark data
    if not os.path.exists('data/gesture_data.csv'):
        print("Error: No data found. Run collection.py first.")
        return

    data = pd.read_csv('data/gesture_data.csv', header=None)
    X = data.iloc[:, :-1].values.astype('float32') # 63 coordinates (21x3)
    y = data.iloc[:, -1].values # Gesture labels

    # Encode labels (e.g., "Palm" to 0, "Fist" to 1)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Save the encoder for the main controller to use
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    # [cite_start]Split for validation to ensure reliability [cite: 61]
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

    # Simple but powerful Dense Neural Network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        [cite_start]tf.keras.layers.Dropout(0.2), # Error prevention [cite: 52]
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Retraining model... please wait.")
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)
    
    # [cite_start]Save the final model [cite: 75]
    model.save('models/gesture_model.h5')
    print("Success! Model saved in models/gesture_model.h5")

if __name__ == "__main__":
    retrain_model()

