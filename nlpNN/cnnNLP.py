from PIL import Image
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
import matplotlib.pyplot as plt
import re

#claude suggested method to use less memory
import gc
gc.collect()
tf.keras.backend.clear_session()

#load the nlp model
import gensim.downloader as api
NLPmodel = api.load("glove-wiki-gigaword-100")  # 100D, ~91MB


#removes index from the end of the word and gets its vector
#alerts user if word is not contained in the model
def getVector(word):
    word = re.sub(r"\d+$", "", word.lower())
    if word in NLPmodel:
        return NLPmodel[word]
    else:
        print(f"Warning: Word '{word}' not found in model vocabulary")
        return np.zeros(100)

#used to measure how similar the predicted vector is from the actual one
def cosine_similarity_loss(y_true, y_pred):
    # Add small epsilon to prevent division by zero
    epsilon = 1e-7
    y_true_norm = K.l2_normalize(y_true, axis=-1)
    y_pred_norm = K.l2_normalize(y_pred, axis=-1)
    cosine = K.sum(y_true_norm * y_pred_norm, axis=-1)
    return 1 - cosine

#initalises where to find the test and train files and labels them
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
                    # participant_words = participant_words[:800] #####?
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    test_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])
                    y_train.extend(getVector(word) for word in participant_words[:split_idx])  #removeIndex gets rid of the word index
                    y_test.extend(getVector(word) for word in participant_words[split_idx:])
        else:
            print(f"Folder not found: {full_path}")

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    print(f"Length of y_train: {len(y_train)}")
    print(f"Length of y_test: {len(y_test)}")
    return train_files, test_files, y_train, y_test

# def load_sample(folder_path):
#     """Loads and processes 32 BMP files into 56x107x1 spectrograms."""
#     sample_data = []
#     files = sorted(os.listdir(folder_path))

#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         image = Image.open(file_path).convert("L")
#         image = image.resize((107,56))
#         image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
#         image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
#         sample_data.append(image_array)
#     if len(sample_data) != 32:
#         print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")

#     # plt.imshow(sample_data[0].squeeze(), cmap='gray')
#     # plt.title(f"Resized Image from {folder_path}")
#     # plt.show()
#     return sample_data  # Returns list of 32 arrays of shape (56, 107, 1)

def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

# def spectrogram_CNN():
#     """Creates a CNN model with optimal parameters from Optuna."""
#     inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
#     cnn_outputs = []
#     for input_layer in inputs:
#         x = Conv2D(filters=40, kernel_size=(7, 5), activation='relu', padding='same')(input_layer)
#         x = MaxPool2D(pool_size=(3, 3))(x)
#         x = Flatten()(x)
#         cnn_outputs.append(x)
#     combined = Concatenate()(cnn_outputs)
#     x = Dense(192, activation='relu')(combined)
#     x = Dropout(0.2628)(x)
#     output = Dense(100, activation='linear')(x)
#     model = Model(inputs=inputs, outputs=output)
#     return model

def spectrogram_CNN():
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(100, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(train_files, test_files, y_train, y_test):
    # Prepare training and testing data
    X_train_samples = [load_sample(file) for file in train_files] 
    X_test_samples = [load_sample(file) for file in test_files] 

    # Convert to list of 32 arrays, each with shape (n_samples, 56, 107, 1)
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]

    # Check for NaNs or infinities
    print("Any NaNs in X_train:", any(np.any(np.isnan(x)) for x in X_train))
    print("Any infs in X_train:", any(np.any(np.isinf(x)) for x in X_train))

    # Create and compile model
    model = spectrogram_CNN()
    model.compile(optimizer=SGD(learning_rate=0.00064), loss=cosine_similarity_loss, metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, batch_size=24, epochs=100, verbose=1, shuffle=True)

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")

    predictions = model.predict(X_test, verbose=0)  

    cosine_similarities = np.sum(predictions * y_test, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(y_test, axis=-1)
    )

    print("\nTest Predictions and Cosine Similarities:")
    for i in range(min(30, len(y_test))):  # Print first 5 samples
        print(f"Sample {i}:")
        print(f"  Predicted Vector: {predictions[i][:5]}...")  # Show first 5 elements
        print(f"  True Vector: {y_test[i][:5]}...")
        print(f"  Cosine Similarity: {cosine_similarities[i]:.4f}")
        print("-" * 50)

    return model


# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
folders_names = ["content", "function"]

# Prepare dataset
train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(train_files, test_files, y_train, y_test)

#get it to show the image after it has been resized to see whats up