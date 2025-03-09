import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

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
    sample_data = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()
        data = data.reshape((12, 8, 1)).astype(np.float32)
        sample_data.append(data)
    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")
    return sample_data

def create_model(trial):
    # Hyperparameters to search
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)  # Number of Conv2D layers per input
    filters = trial.suggest_int('filters', 8, 64, step=8)     # Filters in Conv2D
    kernel_w = trial.suggest_int('kernel_w', 3, 7, step=2)    # Kernel width
    kernel_h = trial.suggest_int('kernel_h', 3, 5, step=2)    # Kernel height
    pool_size = trial.suggest_categorical('pool_size', [2, 3])  # Pooling size
    dense_units = trial.suggest_int('dense_units', 64, 256, step=32)  # Dense layer units
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)      # Dropout rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)  # Learning rate
    optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam'])   # Optimizer
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])        # Batch size

    # Model architecture
    inputs = [Input(shape=(12, 8, 1)) for _ in range(32)]
    cnn_outputs = []
    for inp in inputs:
        x = inp
        # Stack Conv2D layers
        for _ in range(n_conv_layers):
            x = Conv2D(filters, (kernel_w, kernel_h), activation='relu', padding='same')(x)
            x = MaxPool2D((pool_size, pool_size))(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    
    combined = Concatenate()(cnn_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Optimizer
    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model, batch_size

def objective(trial, X_train, X_test, y_train, y_test):
    # Split train into train and validation
    X_train_split = []
    X_val_split = []
    for i in range(32):
        train_split, val_split, y_train_split, y_val = train_test_split(
            X_train[i], y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        X_train_split.append(train_split)
        X_val_split.append(val_split)

    model, batch_size = create_model(trial)
    
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val_split, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )
    
    # Evaluate on test set for final scoring
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc  # Optimize for test accuracy

def main():
    # Paths
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folders_path = os.path.join(current_directory, "spectrogramDataGoated")
    folders_names = ["content", "function"]

    # Prepare dataset
    train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

    # Load data
    X_train_samples = [load_sample(file) for file in train_files]
    X_test_samples = [load_sample(file) for file in test_files]
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]

    print("X_train shapes:", [x.shape for x in X_train])
    print("X_test shapes:", [x.shape for x in X_test])
    print("Any NaNs in X_train:", any(np.any(np.isnan(x)) for x in X_train))

    # Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=50)

    # Best model
    print("Best trial:", study.best_trial.params)
    best_model, best_batch_size = create_model(study.best_trial)
    best_model.fit(X_train, y_train, epochs=100, batch_size=best_batch_size, verbose=1)
    
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the best model
    best_model.save('best_spectrogram_model.keras')
    print("Best model saved as 'best_spectrogram_model.keras'")

if __name__ == "__main__":
    main()