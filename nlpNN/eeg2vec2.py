#extra dense units on input branches

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
# import gensim.downloader as api

import re
from datetime import datetime

# glove_model = api.load("glove-wiki-gigaword-100")

# #load NLP model, not using gensim
def load_glove_model(file_path):
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0] 
            vector = np.array(values[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors

# Load the model from your current directory
NLPmodel = load_glove_model("../glove-wiki-gigaword-100") #before it had no .gz, added .gz to see if it works

# NLPmodel = api.load("glove-wiki-gigaword-100")

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

#extracts all the data from the folders
def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}
    train_files = []
    testValid_files = []
    y_train = []
    y_testValid = []
    train_words = []
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
                    train_words.extend(word for word in participant_words[:split_idx])
                    y_testValid.extend(getVector(word) for word in participant_words[split_idx:])
                    testValid_words.extend(word for word in participant_words[split_idx:])
        else:
            print(f"Folder not found: {full_path}")

    testValid_files, y_testValid, testValid_words = shuffle(testValid_files, y_testValid, testValid_words)

    val_idx = int(len(testValid_files) * 0.5)

    X_train_samples = [load_sample(file) for file in train_files] 
    X_test_samples = [load_sample(testValid_files[i]) for i in range(0, val_idx)]
    X_valid_samples = [load_sample(testValid_files[i]) for i in range(val_idx, len(testValid_files))]
    test_words = testValid_words[:val_idx]
    valid_words = testValid_words[val_idx:]
    print ("x test words", len(test_words),":",test_words)
    np.savez_compressed("words_labels.npz", 
                    train_words=train_words,
                    valid_words=valid_words,
                    test_words=test_words)

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

    return X_train, X_test, X_valid, y_train, y_test, y_valid, test_words

#loads all the data required for each word (no EEGs, width, height, grayscale)
def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

#all these hyperparameters were found by optuna
def spectrogram_CNN():
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(3, 3))(x)
        x = Flatten()(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.14646860733325393)(x)

        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.253325747247369)(x)
    output = Dense(100, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def NN(X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words):
    np.savez(os.path.join(saveFolder, "test_data.npz"), 
            X_test=X_test,  
            y_test=y_test)
    np.savez(os.path.join(saveFolder, "train_data.npz"), 
            X_train=X_train, 
            y_train=y_train)
    np.savez(os.path.join(saveFolder, "valid_data.npz"), 
            X_valid=X_valid,  
            y_valid=y_valid)

    #create and compile model
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

    #train model
    model.fit(
        X_train, y_train,
        batch_size=4,
        epochs=40,
        verbose=1,
        shuffle=True,
        callbacks=[saveModelCallback, early_stopping],
        validation_data=(X_valid, y_valid)
    )

    #evaluate model
    test_loss, test_acc, test_cosine_sim = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")
    with open(os.path.join(saveFolder, "results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Cosine Similarity: {test_cosine_sim}\n")

    output_file = "ranking_results.txt"

    #get the preductions of X_test
    predictions = model.predict(X_test, verbose=0)

    #all of the following is displaying and saving info from the model

    #get all cosine similarities between the predictions and y_test
    cosine_similarities = np.sum(predictions * y_test, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(y_test, axis=-1)
    )

    # Write all output to file
    with open(output_file, "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Cosine Similarity: {test_cosine_sim}\n")

        #show predictions vs true to check its not just printing the same vector every time
        f.write("Test Predictions and Cosine Similarities (first 5 samples):\n")
        for i in range(min(5, len(y_test))):
            f.write(f"\nSample {i}: Word = {X_test_words[i]}\n")
            f.write(f"  Predicted Vector: {predictions[i][:45].tolist()}...\n")
            f.write(f"  True Vector: {y_test[i][:45].tolist()}...\n")
            f.write(f"  Cosine Similarity: {cosine_similarities[i]:.4f}\n")
            f.write("-" * 50 + "\n")

        #rankings predicted vectors vs actual vectors
        f.write("\nRanking each prediction against all possible labels:\n")
        ranks = []

        for i in range(len(X_test_words)):
            similarities = []
            for j in range(len(y_test)):
                pred_tensor = tf.constant([predictions[i]], dtype=tf.float32)
                true_tensor = tf.constant([y_test[j]], dtype=tf.float32)
                sim = cosine_similarity(true_tensor, pred_tensor)
                similarities.append(float(sim.numpy()))

            similarities = np.array(similarities)
            sorted_indices = np.argsort(similarities)[::-1]
            correct_index = np.where(sorted_indices == i)[0][0]
            ranks.append(correct_index)

        ranks = np.array(ranks)

        #
        f.write(f"Number of samples: {len(X_test_words)}\n")
        f.write(f"Ranks of correct labels: {ranks.tolist()}\n")
        f.write(f"Mean Reciprocal Rank: {np.mean(1 / (ranks + 1)):.4f}\n")
        f.write(f"Median rank: {np.median(ranks)}\n")
        f.write(f"Top-1 accuracy: {np.mean(ranks == 0):.4f}\n")
        f.write(f"Top-5 accuracy: {np.mean(ranks < 5):.4f}\n")

        # ---------- SIMILARITY MATRIX ----------
        f.write("\nFull Cosine Similarity Matrix and Rank Analysis:\n")
        num_samples = len(y_test)
        similarity_matrix = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(num_samples):
                pred_tensor = tf.constant([predictions[i]], dtype=tf.float32)
                true_tensor = tf.constant([y_test[j]], dtype=tf.float32)
                sim = cosine_similarity(true_tensor, pred_tensor)
                similarity_matrix[i, j] = float(sim.numpy())

        sorted_indices = np.argsort(-similarity_matrix, axis=1)
        matrix_ranks = np.array([np.where(sorted_indices[i] == i)[0][0] for i in range(num_samples)])

        f.write(f"\nRanks of correct labels from matrix: {matrix_ranks.tolist()}\n")
        f.write(f"Mean Reciprocal Rank: {np.mean(1 / (matrix_ranks + 1)):.4f}\n")
        f.write(f"Median Rank: {np.median(matrix_ranks)}\n")
        f.write(f"Top-1 Accuracy: {np.mean(matrix_ranks == 0):.4f}\n")
        f.write(f"Top-5 Accuracy: {np.mean(matrix_ranks < 5):.4f}\n\n")

        #also written in BP version:
        #x_test_words = the words used for testing
        #predictions = the predicted labels
        #y_test = the true labels

        f.write("Cosine Similarity Matrix:\n")
        np.set_printoptions(threshold=np.inf, linewidth=200)
        f.write(np.array2string(similarity_matrix, precision=4, suppress_small=True))

    print(f"All results written to {output_file}")

    # Final return
    return model

# folders_path = os.path.join(current_directory, "../spectrogramDataHighGranFull")
folders_path = "/user/work/dk22310/spectrogramDataHighGranFull2" #dataset path for BP
folders_names = ["content", "function"]

# Prepare dataset
X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
model = NN(X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words)


print ("ended without crashing")
