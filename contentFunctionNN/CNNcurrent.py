import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import SGD

def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}
    train_files = []
    test_files = []
    y_train = []
    y_test = []

    for folder in folders_names:
        full_path = os.path.join(folders_path, folder)
        if os.path.exists(full_path):
            print(f"\nContents of {folder}:")
            participants = os.listdir(full_path)
            for participant in participants:
                participant_path = os.path.join(full_path, participant)
                if os.path.isdir(participant_path):
                    print("Participant:", participant)
                    participant_words = os.listdir(participant_path)
                    random.shuffle(participant_words)
                    participant_words = participant_words[:800]
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    test_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])
                    label = label_map[folder]
                    y_train.extend([label] * split_idx)
                    y_test.extend([label] * (len(participant_words) - split_idx))
        else:
            print(f"Folder not found: {full_path}")

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    print(f"Length of y_train: {len(y_train)}")
    print(f"Length of y_test: {len(y_test)}")
    return train_files, test_files, y_train, y_test

def load_sample(folder_path):
    """Loads and processes 32 CSV files into 12x8x1 spectrograms."""
    sample_data = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()
        # Reshape from (96,) to (12, 8, 1)
        data = data.reshape((12, 8, 1)).astype(np.float32)
        sample_data.append(data)
    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")
    return sample_data  # Returns list of 32 arrays of shape (12, 8, 1)

def spectrogram_CNN():
    """Creates a CNN model with optimal parameters from Optuna."""
    inputs = [Input(shape=(12, 8, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=40, kernel_size=(7, 5), activation='relu', padding='same')(input_layer)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(192, activation='relu')(combined)
    x = Dropout(0.2628)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(train_files, test_files, y_train, y_test):
    # Prepare training and testing data
    X_train_samples = [load_sample(file) for file in train_files]  # List of (n_train, 32, 12, 8, 1)
    X_test_samples = [load_sample(file) for file in test_files]    # List of (n_test, 32, 12, 8, 1)

    # Convert to list of 32 arrays, each with shape (n_samples, 12, 8, 1)
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]

    # Check for NaNs or infinities
    print("Any NaNs in X_train:", any(np.any(np.isnan(x)) for x in X_train))
    print("Any infs in X_train:", any(np.any(np.isinf(x)) for x in X_train))

    # Create and compile model
    model = spectrogram_CNN()
    model.compile(optimizer=SGD(learning_rate=0.00064), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, shuffle=True)

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")

    return model

# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataGoated")
folders_names = ["content", "function"]

# Prepare dataset
train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(train_files, test_files, y_train, y_test)