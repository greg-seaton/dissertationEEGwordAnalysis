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
import optuna
from optuna.integration import TFKerasPruningCallback

#latest version

#memory management dont know if it does anything, but worth a try
import gc
gc.collect()
tf.keras.backend.clear_session()

#setting up model save folder. (x and y test data, a number of models, best test performance)
current_directory = os.path.dirname(os.path.abspath(__file__))
saveFolderName = datetime.now().strftime("%Y-%m-%d_%H-%M")
saveFolder = os.path.join(current_directory, "savedCFmodels")
saveFolder = os.path.join(saveFolder, saveFolderName)
os.makedirs(saveFolder, exist_ok=True)

#optuna file logging
optuna_log_file = os.path.join(saveFolder, "optuna_log.txt")
def log_optuna_trial(trial_number, params, accuracy):
    with open(optuna_log_file, "a") as f:
        f.write(f"Trial {trial_number}:\n")
        f.write(f"  Parameters: {params}\n")
        f.write(f"  Accuracy: {accuracy}\n\n")

#CNN structure with changing hyperparameters
def spectrogram_CNN(trial):
    """Creates a CNN model with parameters from Optuna trial."""
    # Hyperparameters to optimize
    filters = trial.suggest_int('filters', 16, 128, step=8)  # Number of Conv filters
    kernel_size_h = trial.suggest_int('kernel_size_h', 3, 9, step=2)  # Kernel height
    kernel_size_w = trial.suggest_int('kernel_size_w', 3, 9, step=2)  # Kernel width
    pool_size = trial.suggest_int('pool_size', 2, 4)  # Pooling size (no step needed)
    dense_units = trial.suggest_int('dense_units', 64, 512, step=32)  # Dense layer units
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Dropout rate (unchanged)

    # Model structure (similar to original)
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=filters, kernel_size=(kernel_size_h, kernel_size_w), 
                   activation='relu', padding='same')(input_layer)
        x = MaxPool2D(pool_size=(pool_size, pool_size))(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

# Extract raw data from an EEG instance folder
def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

# Goes from folders to raw data to train, test and validate (unchanged)
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

    testValid_files, y_testValid = shuffle(testValid_files, y_testValid)

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

# Modified NN function that accepts a trial for optimization
def objective(trial, X_train, X_valid, y_train, y_valid):
    """Optuna objective function for optimizing the CNN model."""
    # Clear TF session to prevent memory issues between trials
    tf.keras.backend.clear_session()
    
    # Create a trial-specific folder
    trial_folder = os.path.join(saveFolder, f"trial_{trial.number}")
    os.makedirs(trial_folder, exist_ok=True)
    
    # Hyperparameters to optimize
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Optimizer selection
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    if optimizer_name == 'SGD':
        # SGD specific parameters
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        # Adam specific parameters
        beta_1 = trial.suggest_float('beta_1', 0.8, 0.999)
        beta_2 = trial.suggest_float('beta_2', 0.8, 0.999)
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    
    # Batch size
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    
    # Create model checkpoint callback for this trial
    # checkpoint_callback = ModelCheckpoint(
    #     os.path.join(trial_folder, "model_best.keras"),
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     verbose=0
    # )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=7,  # Reduced patience for faster trials
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    
    # Optuna pruning callback to stop unpromising trials early
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
    
    # Create and compile model
    model = spectrogram_CNN(trial)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model with reduced epochs for faster trials
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=20,  # Reduced for optimization, increase for final model
        verbose=0,  # Reduced verbosity for cleaner output
        shuffle=True,
        # callbacks=[checkpoint_callback, early_stopping, pruning_callback],
        callbacks=[early_stopping, pruning_callback],
        validation_data=(X_valid, y_valid)
    )
    
    # Get best validation accuracy
    val_accuracy = max(history.history['val_accuracy'])
    
    # Log the results
    params = {
        'filters': trial.params['filters'],
        'kernel_size': (trial.params['kernel_size_h'], trial.params['kernel_size_w']),
        'pool_size': trial.params['pool_size'],
        'dense_units': trial.params['dense_units'],
        'dropout_rate': trial.params['dropout_rate'],
        'learning_rate': trial.params['learning_rate'],
        'optimizer': optimizer_name,
        'batch_size': trial.params['batch_size']
    }
    
    # Add optimizer-specific parameters
    if optimizer_name == 'SGD':
        params['momentum'] = trial.params['momentum']
    else:  # Adam
        params['beta_1'] = trial.params['beta_1']
        params['beta_2'] = trial.params['beta_2']
    
    log_optuna_trial(trial.number, params, val_accuracy)
    
    return val_accuracy

# Train the final model with best parameters
def train_final_model(study, X_train, X_test, X_valid, y_train, y_test, y_valid):
    """Train the final model using the best parameters found by Optuna."""
    # Create a folder for the final model
    final_model_folder = os.path.join(saveFolder, "final_model")
    os.makedirs(final_model_folder, exist_ok=True)
    
    # Get best parameters
    best_params = study.best_params
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Create model with best parameters
    trial = optuna.trial.FixedTrial(best_params)
    model = spectrogram_CNN(trial)
    
    # Setup optimizer
    if best_params['optimizer'] == 'SGD':
        optimizer = SGD(learning_rate=best_params['learning_rate'], 
                        momentum=best_params['momentum'])
    else:
        optimizer = Adam(learning_rate=best_params['learning_rate'],
                         beta_1=best_params['beta_1'],
                         beta_2=best_params['beta_2'])
    
    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Setup callbacks
    saveModelCallback = ModelCheckpoint(
        os.path.join(final_model_folder, "model_epoch{epoch:02d}_valloss{val_loss:.4f}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    
    # Train model with more epochs
    history = model.fit(
        X_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=40,  # More epochs for final training
        verbose=1,
        shuffle=True,
        callbacks=[saveModelCallback, early_stopping],
        validation_data=(X_valid, y_valid)
    )
    
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")
    
    # Save test results
    with open(os.path.join(final_model_folder, "results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n)")

# Paths
folders_path = os.path.join(current_directory, "../spectrogramDataHighGran")
folders_names = ["content", "function"]

# Prepare dataset
X_train, X_test, X_valid, y_train, y_test, y_valid = NN_prep(folders_path, folders_names)

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=10)

print ("ended without crashing")