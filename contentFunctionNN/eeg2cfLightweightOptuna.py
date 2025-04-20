import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import optuna
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

def create_model(trial):
    # Hyperparameters to optimize
    first_conv_filters = trial.suggest_categorical('first_conv_filters', [16, 32, 64])
    second_conv_filters = trial.suggest_categorical('second_conv_filters', [32, 64, 128])
    
    # Use string representation for kernel size to avoid Optuna warnings
    kernel_size_str = trial.suggest_categorical('kernel_size', ['2x2', '3x3'])
    
    # Convert kernel_size string to tuple for actual use
    if kernel_size_str == '2x2':
        kernel_size = (2, 2)
    else:  # '3x3'
        kernel_size = (3, 3)
        
    dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
    final_dense_units = trial.suggest_categorical('final_dense_units', [128, 192, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    final_dropout_rate = trial.suggest_float('final_dropout_rate', 0.1, 0.5)
    
    # Create model with the suggested hyperparameters
    inputs = [Input(shape=(96,)) for _ in range(32)]
    cnn_outputs = []
    
    for input_layer in inputs:
        # Reshape flat input to 2D image format (assuming 12x8 is the correct dimension)
        x = Reshape((8, 12, 1))(input_layer)
        
        # CNN layers with hyperparameters
        x = Conv2D(filters=first_conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(second_conv_filters, kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        cnn_outputs.append(x)

    combined = Concatenate()(cnn_outputs)
    x = Dense(final_dense_units, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(final_dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    
    # Optimizer hyperparameters
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    if optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial, X_train, X_valid, y_train, y_valid):
    # Clear backend session to avoid memory issues
    tf.keras.backend.clear_session()
    
    # Print trial information 
    print(f"\n{'='*50}")
    print(f"Trial #{trial.number}")
    print(f"{'='*50}")
    
    # Create model with trial hyperparameters
    model = create_model(trial)
    
    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    patience = trial.suggest_int('patience', 5, 15)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    
    # Print current hyperparameters
    print("\nTrial Hyperparameters:")
    for param_name, param_value in trial.params.items():
        print(f"  {param_name}: {param_value}")
    print()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=40,
        verbose=1,
        shuffle=True,
        callbacks=[early_stopping],
        validation_data=(X_valid, y_valid)
    )
    
    # Get validation accuracy from the best epoch
    val_accuracy = max(history.history['val_accuracy'])
    
    # Print trial results
    print(f"\nTrial #{trial.number} Results:")
    print(f"  Best validation accuracy: {val_accuracy:.4f}")
    print(f"{'='*50}\n")
    
    return val_accuracy

def run_optuna_optimization(X_train, X_test, X_valid, y_train, y_test, y_valid, n_trials=15):
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize', 
                               study_name='cnn_optimization',
                               pruner=optuna.pruners.MedianPruner())
    
    try:
        study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), 
                      n_trials=n_trials,
                      show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    
    # Print optimization results
    print("\n" + "="*80)
    print("Hyperparameter Optimization Results")
    print("="*80)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  Validation accuracy: {trial.value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Create and train model with the best hyperparameters
    print("\n" + "="*80)
    print("Training Final Model with Best Hyperparameters")
    print("="*80)
    
    tf.keras.backend.clear_session()
    best_model = create_model(trial)
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=trial.params['patience'],
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
    
    best_model.fit(
        X_train, y_train,
        batch_size=trial.params['batch_size'],
        epochs=40,
        verbose=1,
        shuffle=True,
        callbacks=[early_stopping],
        validation_data=(X_valid, y_valid)
    )
    
    # Evaluate final model on test set
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print("\n" + "="*80)
    print("Final Model Evaluation on Test Set")
    print("="*80)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return best_model, study

if __name__ == "__main__":
    # Paths
    folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataGoated")
    folders_names = ["content", "function"]

    # Prepare dataset
    print("Preparing dataset...")
    X_train, X_test, X_valid, y_train, y_test, y_valid, X_test_words = NN_prep(folders_path, folders_names)

    # Run Optuna hyperparameter optimization
    print("\nStarting hyperparameter optimization with Optuna...")
    best_model, study = run_optuna_optimization(X_train, X_test, X_valid, y_train, y_test, y_valid, n_trials=15)

    print("Optimization completed successfully!")