import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Content and function each contain 50 words. This script is initially designed for one subject.
# Each folder for a different word should be split into different test and train sets.

def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}  # Assign labels

    train_files = []
    test_files = []
    y_train = []
    y_test = []

    for folder in folders_names:  # Loops through "content" and "function"
        full_path = os.path.join(folders_path, folder)

        if os.path.exists(full_path):
            print(f"\nContents of {folder}:")
            participants = os.listdir(full_path)

            for participant in participants:  # Loops through participants
                participant_path = os.path.join(full_path, participant)

                if os.path.isdir(participant_path):
                    print("Participant:", participant)
                    participant_words = os.listdir(participant_path)  # Get all word data from participant

                    # Shuffle the data
                    random.shuffle(participant_words)

                    # Train-test split (80% train, 20% test)
                    split_idx = int(len(participant_words) * 0.8)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    test_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])

                    # Generate labels (0 for content, 1 for function)
                    label = label_map[folder]
                    y_train.extend([label] * split_idx)
                    y_test.extend([label] * (len(participant_words) - split_idx))

        else:
            print(f"Folder not found: {full_path}")

    return train_files, test_files, y_train, y_test


def spectrogram_CNN():
    """CNN model to process the structured spectrogram-like data (32, 96, 1)."""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 96, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model


def load_sample(folder_path):
    """Loads and processes 32 CSV files (1 word) into a (32, 96) NumPy array."""
    sample_data = []
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()  # Load as NumPy array
        sample_data.append(data)  # Each row is a (96,) array

    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")

    return np.array(sample_data)  # Shape (32, 96)


def NN(train_files, test_files, y_train, y_test):
    """Trains the CNN model on spectrogram-like data."""
    model = spectrogram_CNN()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare training data
    X_train, X_test = [], []

    for train_instance in train_files:
        X_train.append(load_sample(train_instance))  # Each is (32, 96)
    for test_instance in test_files:
        X_test.append(load_sample(test_instance))  # Each is (32, 96)

    X_train = np.array(X_train)  # Shape: (num_samples, 32, 96)
    X_test = np.array(X_test)  # Shape: (num_samples, 32, 96)

    # Reshape for CNN input (add channel dimension)
    X_train = X_train[..., np.newaxis]  # Shape: (num_samples, 32, 96, 1)
    X_test = X_test[..., np.newaxis]  # Shape: (num_samples, 32, 96, 1)

    print("Train shape:", X_train.shape)  # Expect (80, 32, 96, 1)
    print("Test shape:", X_test.shape)  # Expect (20, 32, 96, 1)

    # Train the model
    model.fit(X_train, np.array(y_train), epochs=45, batch_size=8, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, np.array(y_test))
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    return model


# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataConverted1channel")
folders_names = ["content", "function"]

# Prepare dataset
train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
NN(train_files, test_files, y_train, y_test)
