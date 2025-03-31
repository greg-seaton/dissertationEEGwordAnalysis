#no extra dense units on input branches

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
from sklearn.utils import shuffle

import re
from datetime import datetime


#claude suggested method to use less memory
import gc
gc.collect()
tf.keras.backend.clear_session()

#load NLP model, not using gensim
def load_glove_model(file_path):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # First token is the word
            vector = np.array(values[1:], dtype=np.float32)  # Rest are vector values
            word_vectors[word] = vector
    return word_vectors

# Load the model from your current directory
NLPmodel = load_glove_model("../glove-wiki-gigaword-100")


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
                    participant_words = os.listdir(participant_path) #all the word folder locations
                    random.shuffle(participant_words)
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]]) #first 70%
                    testValid_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]]) #last 30%
                    y_train.extend(getVector(word) for word in participant_words[:split_idx])
                    y_testValid.extend(getVector(word) for word in participant_words[split_idx:])
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
        x = MaxPool2D(pool_size=(3, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)  # Consider replacing this with GlobalAveragePooling2D()
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(512, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.47479559089385054)(x)  # Increased dropout
    output = Dense(100, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words):
    np.savez(os.path.join(saveFolder, "test_data.npz"), 
            X_test=X_test,  # Convert to a single array
            y_test=y_test)

    # Create and compile model
    model = spectrogram_CNN()
    model.compile(optimizer=Adam(learning_rate=0.003484660925656064, beta_1=0.886221166624916, beta_2=0.9059015852303618), loss=cosine_similarity_loss, metrics=['accuracy', cosine_similarity])

    #implementing early stopping
    early_stopping = EarlyStopping(
        monitor="val_accuracy",  # Monitor validation loss
        patience=10,         # Stop if val_loss does not improve for 10 epochs
        restore_best_weights=True,  # Restore model weights from best epoch
        mode="max",
        verbose=1
    )

    # Train model
    model.fit(
        X_train, y_train,
        batch_size=4,
        epochs=40,
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



    ##visualise the predictions
    predictions = model.predict(X_test, verbose=0)

    cosine_similarities = np.sum(predictions * y_test, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(y_test, axis=-1)
    )

    print("\nTest Predictions and Cosine Similarities:")
    for i in range(min(5, len(y_test))):
        print(f"Sample {i}:")
        print(f"  Predicted Vector: {predictions[i][:45]}...")
        print(f"  True Vector: {y_test[i][:45]}...")
        print(f"  Cosine Similarity: {cosine_similarities[i]:.4f}")
        print("-" * 50)

    #post analysis, ranking prediction against actual labels
    ranks = []  # To store the ranks of the correct labels

    for i in range(len(X_test_words)):
        similarities = []
        for j in range(len(y_test)):
            # Convert numpy arrays to TensorFlow tensors
            pred_tensor = tf.constant([predictions[i]], dtype=tf.float32)
            true_tensor = tf.constant([y_test[j]], dtype=tf.float32)
            
            # Use your cosine_similarity function with TensorFlow tensors
            sim = cosine_similarity(true_tensor, pred_tensor)
            
            # Execute the TensorFlow operation and get the result as a numpy value
            sim_value = float(sim.numpy())
            similarities.append(sim_value)
        
        # Convert to numpy array
        similarities = np.array(similarities)
        
        # Sort indices in descending order (highest similarity first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Find the position of the correct index (which is i)
        correct_index = np.where(sorted_indices == i)[0][0]
        
        # Add the rank to our list
        ranks.append(correct_index)

    print ("Number of samples", len(X_test_words))
    print("Ranks of correct labels:", ranks)
    print(f"Mean Reciprocal Rank: {np.mean(1 / (np.array(ranks) + 1))}")
    print(f"Median rank: {np.median(ranks)}")
    print(f"Top-1 accuracy: {sum(np.array(ranks) == 0) / len(ranks)}")
    print(f"Top-5 accuracy: {sum(np.array(ranks) < 5) / len(ranks)}")



    #predictions - predicted vectors
    #y_test - true vectors
    #X_test_words - contains the word

    return model

folders_path = os.path.join(current_directory, "../spectrogramDataHighGran")
folders_names = ["content", "function"]

# Prepare dataset
X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words)


print ("ended without crashing")
