import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
import optuna
from sklearn.model_selection import train_test_split


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


    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return train_files, test_files, y_train, y_test


def spectrogram_CNN():
    """Creates a CNN model that processes 32 individual spectrograms separately before concatenation."""
    inputs = [Input(shape=(96, 1)) for _ in range(32)]  # 32 independent spectrograms

    eeg_cnn_elements = []
    for input_layer in inputs:
        # Use the custom layer for dimension expansion
        x = Conv1D(3, 5, activation='relu', padding='same')(input_layer)
        x = MaxPool1D(3, strides=3)(x)
        x = Flatten()(x)
        eeg_cnn_elements.append(x)



    # Combine all CNN outputs
    combined = Concatenate()(eeg_cnn_elements)

    # Fully connected layers ChatGPT choice
    # x = Dense(64, activation='relu')(combined)
    # x = Dropout(0.3)(x)
    # x = Dense(32, activation='relu')(x)
    # output = Dense(1, activation='sigmoid')(x)  # Binary classification

    # model = Model(inputs=inputs, outputs=output)

    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)



    #trying to copy Klaudia
    # output = Dense(1, activation='sigmoid')(x)  # Binary classification
    # # output = Dense(units=2, activation='softmax')(combined)


    model = Model(inputs=inputs, outputs=output)


    return model



def load_sample(folder_path):
    """Loads and processes 32 CSV files (1 word) into a list of (96,) NumPy arrays."""
    sample_data = []
    files = sorted(os.listdir(folder_path))  # Ensure consistent ordering

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()  # Load as NumPy array
        sample_data.append(data)  # Each row is a (96,) array

    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")

    return sample_data  # Returns list of 32 arrays of shape (96,)


def create_model(trial):
    """Creates a CNN model with hyperparameters suggested by Optuna"""
    # Hyperparameter search space
    conv_filters = trial.suggest_categorical('conv_filters', [16, 32, 64])
    kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)  # 3, 5, or 7
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    
    # Model architecture
    inputs = [Input(shape=(96, 1)) for _ in range(32)]
    
    conv_outputs = []
    for inp in inputs:
        x = Conv1D(conv_filters, kernel_size, activation='relu', padding='same')(inp)
        x = MaxPool1D(3, strides=3)(x)
        x = Flatten()(x)
        conv_outputs.append(x)
    
    combined = Concatenate()(conv_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Optimizer configuration
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function"""
    # Create model with suggested hyperparameters
    model = create_model(trial)
    
    # Training configuration
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 30, 100, step=10)
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return best validation accuracy
    return max(history.history['val_accuracy'])

# Modified NN function with Optuna integration
def NN(train_files, test_files, y_train, y_test):
    # Prepare and split data
    X_train = [load_sample(file) for file in train_files]
    X_test = [load_sample(file) for file in test_files]
    
    # Convert to numpy arrays and reshape
    X_train = [np.array([x[i] for x in X_train]).reshape(-1, 96, 1) for i in range(32)]
    X_test = [np.array([x[i] for x in X_test]).reshape(-1, 96, 1) for i in range(32)]
    
    # Create validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        list(zip(*X_train)),  # Zip 32 inputs together
        y_train,
        test_size=0.2,
        random_state=42
    )
    
    # Unzip back to 32 inputs
    X_train_split = [np.array([x[i] for x in X_train_split]) for i in range(32)]
    X_val = [np.array([x[i] for x in X_val]) for i in range(32)]
    
    # Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_split, X_val, y_train_split, y_val),
                   n_trials=30,
                   show_progress_bar=True)
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    final_model = create_model(study.best_trial)
    final_model.fit(
        X_train,
        y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = final_model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    return final_model


# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataConverted1channel")
folders_names = ["content", "function"]

# Prepare dataset
train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Train and evaluate CNN
NN(train_files, test_files, y_train, y_test)



##terminal:

# [I 2025-02-12 14:25:57,202] A new study created in memory with name: no-name-dc37f90b-ea21-40e0-8752-ed945078dc00
#   0%|                                                                   | 0/30 [00:00<?, ?it/s]2025-02-12 14:25:57.216437: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
# [I 2025-02-12 14:26:08,842] Trial 0 finished with value: 0.6875 and parameters: {'conv_filters': 64, 'kernel_size': 5, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 0.0004978749271347723, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 90}. Best is trial 0 with value: 0.6875.
# [I 2025-02-12 14:26:16,162] Trial 1 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0071277787115917625, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 80}. Best is trial 0 with value: 0.6875.
# [I 2025-02-12 14:26:22,171] Trial 2 finished with value: 0.6875 and parameters: {'conv_filters': 64, 'kernel_size': 3, 'dense_units': 64, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0001307266039826665, 'optimizer': 'sgd', 'batch_size': 64, 'epochs': 90}. Best is trial 0 with value: 0.6875.
# [I 2025-02-12 14:26:32,099] Trial 3 finished with value: 0.75 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0066403600892251365, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 100}. Best is trial 3 with value: 0.75.
# [I 2025-02-12 14:26:37,070] Trial 4 finished with value: 0.8125 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0006415904587837178, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:26:51,567] Trial 5 finished with value: 0.75 and parameters: {'conv_filters': 64, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0006676254494005231, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:26:57,131] Trial 6 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0008668602170829665, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 60}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:02,506] Trial 7 finished with value: 0.5 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 0.007359759655377647, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 70}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:09,352] Trial 8 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.0006022241190554424, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 30}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:15,063] Trial 9 finished with value: 0.8125 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.005197248929634005, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:24,077] Trial 10 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 7, 'dense_units': 256, 'dropout_rate': 0.2, 'learning_rate': 0.002227854946581981, 'optimizer': 'adam', 'batch_size': 128, 'epochs': 60}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:29,907] Trial 11 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0024160263173176365, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:37,527] Trial 12 finished with value: 0.75 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.00023148276116101885, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 50}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:45,213] Trial 13 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0022329430740813416, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 30}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:51,276] Trial 14 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.00029350509164010854, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 50}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:27:58,195] Trial 15 finished with value: 0.75 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 0.0013840727199504817, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:05,009] Trial 16 finished with value: 0.75 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0036530241101427998, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 50}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:12,272] Trial 17 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 3, 'dense_units': 128, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0012355609503997042, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 30}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:21,429] Trial 18 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.0042144021074809815, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 70}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:34,554] Trial 19 finished with value: 0.6875 and parameters: {'conv_filters': 64, 'kernel_size': 5, 'dense_units': 64, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.00039334816722777987, 'optimizer': 'sgd', 'batch_size': 64, 'epochs': 40}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:39,673] Trial 20 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 0.00012105017000712184, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 50}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:49,167] Trial 21 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.005429800435945272, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 80}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:28:57,970] Trial 22 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.009697103299810675, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 100}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:09,121] Trial 23 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.2, 'learning_rate': 0.003921665563408985, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 100}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:17,935] Trial 24 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 3, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0017506279256775498, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 60}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:27,850] Trial 25 finished with value: 0.8125 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0009463294421998681, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 80}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:37,481] Trial 26 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0009208709478733293, 'optimizer': 'adam', 'batch_size': 32, 'epochs': 80}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:44,300] Trial 27 finished with value: 0.6875 and parameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0002293493378694494, 'optimizer': 'sgd', 'batch_size': 32, 'epochs': 70}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:29:55,365] Trial 28 finished with value: 0.75 and parameters: {'conv_filters': 64, 'kernel_size': 5, 'dense_units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.00039346282476858076, 'optimizer': 'adam', 'batch_size': 64, 'epochs': 90}. Best is trial 4 with value: 0.8125.
# [I 2025-02-12 14:30:01,170] Trial 29 finished with value: 0.6875 and parameters: {'conv_filters': 32, 'kernel_size': 7, 'dense_units': 128, 'dropout_rate': 0.5, 'learning_rate': 0.0007202785638519901, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 30}. Best is trial 4 with value: 0.8125.
# Best trial: 4. Best value: 0.8125: 100%|███████████████████████| 30/30 [04:03<00:00,  8.13s/it]
# Best hyperparameters: {'conv_filters': 16, 'kernel_size': 5, 'dense_units': 256, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0006415904587837178, 'optimizer': 'sgd', 'batch_size': 128, 'epochs': 40}


# {
#     'conv_filters': 16,
#     'kernel_size': 5,
#     'dense_units': 256,
#     'dropout_rate': 0.3,
#     'learning_rate': 0.0006415904587837178,
#     'optimizer': 'sgd',
#     'batch_size': 128,
#     'epochs': 40
# }