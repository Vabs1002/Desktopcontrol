
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def run_retraining():
    print("--- Starting One-Click Retraining ---")
    
    # 1. Load the data captured by collection.py
    csv_path = 'data/gesture_data.csv'
    if not os.path.exists(csv_path):
        print("Error: No dataset found at 'data/gesture_data.csv'. Run collection.py first!")
        return

    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values.astype('float32') 
    y = df.iloc[:, -1].values 

    # 2. Convert text labels into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the encoder so main.py knows what the numbers mean later
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # 3. Split data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 4. Build the Neural Network Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)), 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax') 
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # 5. Train the model
    print("Training in progress... please wait.")
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # 6. Save the trained model
    model.save('models/gesture_model.h5')
    
    print("\n--- Retraining Complete! ---")
    print("Model saved as 'models/gesture_model.h5'")

if __name__ == "__main__":
    run_retraining()
