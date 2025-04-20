import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

current_directory = os.path.dirname(os.path.abspath(__file__))

def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}
    train_files = []
    testValid_files = []
    y_train = []
    y_testValid = []
    testValid_words = []

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
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    testValid_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])
                    label = label_map[folder]
                    y_train.extend([label] * split_idx)
                    y_testValid.extend([label] * (len(participant_words) - split_idx))
                    testValid_words.extend(word for word in participant_words[split_idx:])

        else:
            print(f"Folder not found: {full_path}")

    testValid_files, y_testValid, testValid_words = shuffle(testValid_files, y_testValid, testValid_words)

    val_idx = int(len(testValid_files) * 0.5)

    X_train_samples = [load_sample(file) for file in train_files] 
    X_test_samples = [load_sample(testValid_files[i]) for i in range(0, val_idx)]
    X_valid_samples = [load_sample(testValid_files[i]) for i in range(val_idx, len(testValid_files))]
    X_test_words = testValid_words[:val_idx]

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

    return X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words

def load_sample(folder_path):
    sample_data = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()
        sample_data.append(data)
    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")
    return sample_data

def spectrogram_CNN():
    # Option 1: Adapt the model to use 1D data directly
    inputs = [Input(shape=(96,)) for _ in range(32)]
    cnn_outputs = []
    
    for input_layer in inputs:
        # Reshape flat input to 2D image format (assuming 12x8 is the correct dimension)
        x = Reshape((8, 12, 1))(input_layer)
        
        # CNN layers adapted for smaller input size
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        cnn_outputs.append(x)

    combined = Concatenate()(cnn_outputs)
    x = Dense(192, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.2628)(x) 
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(X_train, X_test, X_valid, y_train, y_test, y_valid):

    # #verify data shapes
    # print(f"X_train shape: {X_train[0].shape}, y_train shape: {len(y_train)}")
    # print(f"X_valid shape: {X_valid[0].shape}, y_valid shape: {len(y_valid)}")
    # print(f"X_test shape: {X_test[0].shape}, y_test shape: {len(y_test)}")

    # print ("")
    # print("Class distribution in y_test:", np.unique(y_test, return_counts=True))
    # print("Class distribution in y_valid:", np.unique(y_valid, return_counts=True))

    #save data to file so that it can be retested at a later time
    # np.savez(os.path.join(saveFolder, "test_data.npz"), 
    #         X_test=X_test,
    #         y_test=y_test)
    
    # Create and compile model

    early_stopping = EarlyStopping(
        monitor="val_accuracy",  # Monitor validation loss
        patience=10,         # Stop if val_loss does not improve for 10 epochs
        restore_best_weights=True,  # Restore model weights from best epoch
        mode="max",
        verbose=1
    )

    model = spectrogram_CNN()
    model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=40,
        verbose=1,
        shuffle=True,
        callbacks=[early_stopping],
        validation_data=(X_valid, y_valid)
    )
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")
    # with open(os.path.join(saveFolder, "results.txt"), "w") as f:
    #     f.write(f"Test loss: {test_loss}\n")
    #     f.write(f"Test Accuracy: {test_acc}\n")

    return model

# Paths
folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataGoated")
folders_names = ["content", "function"]

# Prepare dataset
X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(X_train, X_test, X_valid, y_train, y_test, y_valid)

print ("ended without crashing")

