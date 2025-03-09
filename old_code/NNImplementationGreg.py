import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Content and function each contain 50 words. This script is initially designed for one subject.
# Each folder for a different word should be split into different test and train sets.


##
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

                    # Train-test split
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




def csv_nn():
    """Builds a small neural network to process each CSV file (96 inputs)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Outputs a single value per CSV
    ])
    return model

def word_nn():
    """Builds a neural network that takes 32 outputs from first-level networks as input."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Final classification (content vs function)
    ])
    return model


def load_sample(folder_path):
    """Loads and processes 32 CSV files (1 word) into a (32, 96) NumPy array."""
    sample_data = []
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path,file)
        data = pd.read_csv(file_path, header=None).values.flatten()  # Load as NumPy array
        sample_data.append(data)  # Each row is a (96,) array
    
    
    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")

    return np.array(sample_data)  # Shape (32, 96)


#get the first_level_nn working correctly, its just random atm
def NN(train_files, test_files, y_train, y_test):
    # Build the first-level and second-level models
    first_level_nn = csv_nn()
    second_level_nn = word_nn()

    # Compile models
    first_level_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    second_level_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare training data
    X_train, X_test = [], []
    
    for train_instance in train_files:
        X_train.append(load_sample(train_instance))  # Each is (32, 96)
    for test_instance in test_files:
        X_test.append(load_sample(test_instance))  # Each is (32, 96)
    
    X_train = np.array(X_train)  # Shape: (num_samples, 32, 96)
    X_test = np.array(X_test)  # Shape: (num_samples, 32, 96)

    print (X_train.shape) #(80,32,96)
    print (X_test.shape) #(20,32,96)
    
    # Step 1: Train first-level neural networks
    first_level_outputs_train = np.array([first_level_nn.predict(x) for x in X_train])  # Shape (num_samples, 32, 1)
    first_level_outputs_test = np.array([first_level_nn.predict(x) for x in X_test])  # Shape (num_samples, 32, 1)

    first_level_outputs_train = first_level_outputs_train.squeeze(-1)  # Reshape to (num_samples, 32)
    first_level_outputs_test = first_level_outputs_test.squeeze(-1)  # Reshape to (num_samples, 32)

    # Step 2: Train second-level neural network
    second_level_nn.fit(first_level_outputs_train, np.array(y_train), epochs=100, batch_size=8, validation_split=0.2)

    # Evaluate model
    test_loss, test_acc = second_level_nn.evaluate(first_level_outputs_test, np.array(y_test))
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    return second_level_nn  # Return trained model


current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataConverted1channel")
folders_names = ["content", "function"]
                #content = 0, function = 1


train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

NN(train_files, test_files, y_train, y_test)

print (len(train_files))
print (len(test_files))
print (len (y_train), y_train)
print (len (y_test), y_test)





