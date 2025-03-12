from PIL import Image
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import gc
gc.collect()
tf.keras.backend.clear_session()

#setup folders
current_directory = os.path.dirname(os.path.abspath(__file__))
saveFolderName = datetime.now().strftime("%Y-%m-%d_%H-%M")
saveFolder = os.path.join(current_directory, "savedCFmodels")
saveFolder = os.path.join(saveFolder, saveFolderName)

os.makedirs(saveFolder, exist_ok=True)

saveModelCallback = ModelCheckpoint(
    os.path.join(saveFolder, "model_epoch{epoch:02d}_valloss{val_loss:.4f}.keras"),  
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

#declares CNN structure
def spectrogram_CNN():
    """Creates a CNN model with optimal parameters from Optuna."""
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
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

#extrats raw data from an EEG instance folder
def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

#goes from folders to raw data to train,test and validate
def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}
    train_files = []
    testValid_files = []
    y_train = []
    y_testValid = []

    for folder in folders_names:
        full_path = os.path.join(folders_path, folder)
        if os.path.exists(full_path):
            print(f"\nContents of {folder}:")
            participants = os.listdir(full_path)
            for participant in participants:
                participant_path = os.path.join(full_path, participant)
                if os.path.isdir(participant_path):
                    print("Participant:", participant)
                    participant_words = os.listdir(participant_path) #all the word folder locations
                    random.shuffle(participant_words)
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]]) #first 70%
                    testValid_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]]) #last 30%
                    label = label_map[folder]
                    y_train.extend([label] * split_idx)
                    y_testValid.extend([label] * (len(participant_words) - split_idx))
        else:
            print(f"Folder not found: {full_path}")

    testValid_files, y_testValid = shuffle(testValid_files, y_testValid, random_state=69)


    val_idx = int(len(testValid_files) * 0.5)

    X_train_samples = [load_sample(file) for file in train_files] 
    X_test_samples = [load_sample(testValid_files[i]) for i in range(0, val_idx)]
    X_valid_samples = [load_sample(testValid_files[i]) for i in range(val_idx, len(testValid_files))]

    # Convert to list of 32 arrays, each with shape (n_samples, 56, 107, 1)
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]
    X_valid = [np.array([sample[i] for sample in X_valid_samples]) for i in range(32)]

    # Ensure labels match the split of test files

    y_test_split = np.array(y_testValid)
    y_test = y_test_split[:val_idx]
    y_valid = y_test_split[val_idx:]

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    y_valid = np.array(y_valid, dtype=np.float32)


    return X_train, X_test, X_valid, y_train, y_test, y_valid

def NN(X_train, X_test, X_valid, y_train, y_test, y_valid):

    #verify data shapes
    print(f"X_train shape: {X_train[0].shape}, y_train shape: {len(y_train)}")
    print(f"X_valid shape: {X_valid[0].shape}, y_valid shape: {len(y_valid)}")
    print(f"X_test shape: {X_test[0].shape}, y_test shape: {len(y_test)}")

    print ("")
    print("Class distribution in y_test:", np.unique(y_test, return_counts=True))
    print("Class distribution in y_valid:", np.unique(y_valid, return_counts=True))

    #save data to file so that it can be retested at a later time
    np.savez(os.path.join(saveFolder, "test_data.npz"), 
            X_test=X_test,
            y_test=y_test)
    
    # Create and compile model
    model = spectrogram_CNN()
    model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    #implementing early stopping
    early_stopping = EarlyStopping(
        monitor="val_accuracy",  # Monitor validation accuracy
        patience=10,             # Stop if val_accuracy does not improve for 10 epochs
        restore_best_weights=False,  # can experiemnt with this
        mode="max",              # Since higher accuracy is better
        verbose=1
    )

    # Train model
    model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=40,
        verbose=1,
        shuffle=True,
        callbacks=[saveModelCallback, early_stopping],
        validation_data=(X_valid, y_valid)
    )
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")
    with open(os.path.join(saveFolder, "results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    return model

# Paths
folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
folders_names = ["content", "function"]

# Prepare dataset
X_train, X_test, X_valid, y_train, y_test, y_valid = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(X_train, X_test, X_valid, y_train, y_test, y_valid)

print ("ended without crashing")

#get it to show the image after it has been resized to see whats up