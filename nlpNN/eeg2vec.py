from PIL import Image
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.functional as F
from datetime import datetime


#claude suggested method to use less memory
import gc
gc.collect()
tf.keras.backend.clear_session()

#load the nlp model
import gensim.downloader as api
NLPmodel = api.load("glove-wiki-gigaword-100")  # 100D, ~91MB


#setup folders
current_directory = os.path.dirname(os.path.abspath(__file__))
saveFolderName = datetime.now().strftime("%Y-%m-%d_%H-%M")
saveFolder = os.path.join(current_directory, "savedNLPmodels")
saveFolder = os.path.join(saveFolder, saveFolderName)

os.makedirs(saveFolder, exist_ok=True)


##saves the weights of the model so progress is not lost if crashing early
saveModelCallback = ModelCheckpoint(
    os.path.join(saveFolder, "model_epoch{epoch:02d}_valloss{val_loss:.4f}.keras"),  
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

#removes index from the end of the word and gets its vector
#alerts user if word is not contained in the model
#also normalises the vector
def getVector(word):
    word = re.sub(r"\d+$", "", word.lower())
    if word in NLPmodel:
        vector = NLPmodel[word]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    else:
        print(f"Warning: Word '{word}' not found in model vocabulary")
        return np.zeros(100)
    

def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

def cosine_similarity(y_true, y_pred):
    return K.sum(y_true * y_pred, axis=-1) / (K.sqrt(K.sum(y_true**2, axis=-1)) * K.sqrt(K.sum(y_pred**2, axis=-1)))

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
    return train_files, test_files, y_train, y_test

#loads all the data required for each word (no EEGs, width, height, grayscale)
def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data


def spectrogram_CNN():
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)  # Consider replacing this with GlobalAveragePooling2D()
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Increased dropout
    output = Dense(100, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(train_files, test_files, y_train, y_test):
    # Making validation set
    val_idx = int(len(test_files) * 0.5)

    X_train_samples = [load_sample(file) for file in train_files] 
    X_test_samples = [load_sample(test_files[i]) for i in range(0, val_idx)]
    X_valid_samples = [load_sample(test_files[i]) for i in range(val_idx, len(test_files))]

    # Convert to list of 32 arrays, each with shape (n_samples, 56, 107, 1)
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]
    X_valid = [np.array([sample[i] for sample in X_valid_samples]) for i in range(32)]

    # Ensure labels match the split of test files

    y_test_split = np.array(y_test)  # Convert to numpy array for indexing consistency
    y_test = y_test_split[:val_idx]   # First half - for X_test
    y_valid = y_test_split[val_idx:]

    # Debugging shape mismatches
    print(f"X_train shape: {X_train[0].shape}, y_train shape: {len(y_train)}")
    print(f"X_valid shape: {X_valid[0].shape}, y_valid shape: {len(y_valid)}")
    print(f"X_test shape: {X_test[0].shape}, y_test shape: {len(y_test)}")

    np.savez(os.path.join(saveFolder, "test_data.npz"), 
            X_test=X_test,  # Convert to a single array
            y_test=y_test)

    # Create and compile model
    model = spectrogram_CNN()
    model.compile(optimizer=Adam(learning_rate=0.00064), loss=cosine_similarity_loss, metrics=['accuracy', cosine_similarity])

    #implementing early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=10,         # Stop if val_loss does not improve for 10 epochs
        restore_best_weights=True,  # Restore model weights from best epoch
        verbose=1
    )

    # Train model
    model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=2,
        verbose=1,
        shuffle=True,
        callbacks=[saveModelCallback, early_stopping],
        validation_data=(X_valid, y_valid)
    )

    # Evaluate model
    test_loss, test_acc, test_cosine_sim = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")
    with open(os.path.join(saveFolder, "results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Cosine Similarity: {test_cosine_sim}\n")


    predictions = model.predict(X_test, verbose=0)

    cosine_similarities = np.sum(predictions * y_test, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(y_test, axis=-1)
    )

    print("\nTest Predictions and Cosine Similarities:")
    for i in range(min(30, len(y_test))):
        print(f"Sample {i}:")
        print(f"  Predicted Vector: {predictions[i][:45]}...")
        print(f"  True Vector: {y_test[i][:45]}...")
        print(f"  Cosine Similarity: {cosine_similarities[i]:.4f}")
        print("-" * 50)

    return model

folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
folders_names = ["content", "function"]

# Prepare dataset
train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(train_files, test_files, y_train, y_test)


print ("ended without crashing")

#saved results
#sgd test loss (only 40 epochs)
#Test loss: 0.8670620322227478, Test Accuracy: 0.16049382090568542