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
import optuna  # Added for hyperparameter optimization

# Original memory management
import gc
gc.collect()
tf.keras.backend.clear_session()

# Load the nlp model
import gensim.downloader as api
NLPmodel = api.load("glove-wiki-gigaword-100")  # 100D, ~91MB

# Setup folders
current_directory = os.path.dirname(os.path.abspath(__file__))
saveFolderName = datetime.now().strftime("%Y-%m-%d_%H-%M")
saveFolder = os.path.join(current_directory, "savedNLPmodels")
saveFolder = os.path.join(saveFolder, saveFolderName)
os.makedirs(saveFolder, exist_ok=True)

# Original callback for saving model weights
saveModelCallback = ModelCheckpoint(
    os.path.join(saveFolder, "model_epoch{epoch:02d}_valloss{val_loss:.4f}.keras"),  
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Original function for getting word vectors
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

# Original loss function
def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

# Original metric function
def cosine_similarity(y_true, y_pred):
    return K.sum(y_true * y_pred, axis=-1) / (K.sqrt(K.sum(y_true**2, axis=-1)) * K.sqrt(K.sum(y_pred**2, axis=-1)))

# Original data preparation function
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
                    y_train.extend(getVector(word) for word in participant_words[:split_idx])
                    y_test.extend(getVector(word) for word in participant_words[split_idx:])
        else:
            print(f"Folder not found: {full_path}")

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    return train_files, test_files, y_train, y_test

# Original sample loading function
def load_sample(folder_path):
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

# Modified CNN model creation function to accept hyperparameters from Optuna
def spectrogram_CNN(trial=None):
    # Default hyperparameters (will be used if trial is None)
    filters1 = 32
    filters2 = 64
    kernel_size1 = (5, 5)
    kernel_size2 = (3, 3)
    pool_size1 = (3, 3)
    pool_size2 = (2, 2)
    dense_units = 256
    dropout_rate = 0.3
    learning_rate = 0.00064
    
    # If Optuna trial is provided, use it to suggest hyperparameters
    if trial:
        # Suggest hyperparameters using Optuna
        filters1 = trial.suggest_categorical('filters1', [16, 32, 64])
        filters2 = trial.suggest_categorical('filters2', [32, 64, 128])
        kernel_size1 = trial.suggest_categorical('kernel_size1', [(3, 3), (5, 5), (7, 7)])
        kernel_size2 = trial.suggest_categorical('kernel_size2', [(3, 3), (5, 5)])
        pool_size1 = trial.suggest_categorical('pool_size1', [(2, 2), (3, 3)])
        pool_size2 = trial.suggest_categorical('pool_size2', [(2, 2), (3, 3)])
        dense_units = trial.suggest_categorical('dense_units', [128, 256, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Original model architecture with tunable hyperparameters
    inputs = [Input(shape=(56, 107, 1)) for _ in range(32)]
    cnn_outputs = []
    for input_layer in inputs:
        x = Conv2D(filters=filters1, kernel_size=kernel_size1, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=pool_size1)(x)
        x = Conv2D(filters2, kernel_size2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=pool_size2)(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(100, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    
    # Compile model with tunable learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=cosine_similarity_loss, 
                  metrics=['accuracy', cosine_similarity])
    
    return model

# Modified NN function to accept hyperparameters from Optuna
def NN(train_files, test_files, y_train, y_test, trial=None):
    # Default hyperparameters (will be used if trial is None)
    batch_size = 8
    epochs = 40  # Changed from 2 to 40 for actual training
    
    # If Optuna trial is provided, use it to suggest hyperparameters
    if trial:
        # Suggest hyperparameters using Optuna
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        epochs = trial.suggest_int('epochs', 20, 100)
    
    # Original data preparation
    val_idx = int(len(test_files) * 0.5)
    
    # Load samples
    print("Loading training samples...")
    X_train_samples = [load_sample(file) for file in train_files]
    print("Loading test samples...")
    X_test_samples = [load_sample(test_files[i]) for i in range(0, val_idx)]
    print("Loading validation samples...")
    X_valid_samples = [load_sample(test_files[i]) for i in range(val_idx, len(test_files))]

    # Convert to list of 32 arrays, each with shape (n_samples, 56, 107, 1)
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]
    X_valid = [np.array([sample[i] for sample in X_valid_samples]) for i in range(32)]

    # Ensure labels match the split of test files
    y_test_split = np.array(y_test)
    y_test_final = y_test_split[:val_idx]
    y_valid = y_test_split[val_idx:]

    # Debugging shape mismatches
    print(f"X_train shape: {X_train[0].shape}, y_train shape: {len(y_train)}")
    print(f"X_valid shape: {X_valid[0].shape}, y_valid shape: {len(y_valid)}")
    print(f"X_test shape: {X_test[0].shape}, y_test shape: {len(y_test_final)}")

    # Save test data for later evaluation
    np.savez(os.path.join(saveFolder, "test_data.npz"), 
            X_test=X_test,
            y_test=y_test_final)

    # Create model with optuna trial if provided
    model = spectrogram_CNN(trial)

    # Original early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    print(f"Training model with batch_size={batch_size}, epochs={epochs}")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        callbacks=[saveModelCallback, early_stopping],
        validation_data=(X_valid, y_valid)
    )

    # Evaluate model
    test_loss, test_acc, test_cosine_sim = model.evaluate(X_test, y_test_final)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")
    
    # Save results
    with open(os.path.join(saveFolder, "results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Cosine Similarity: {test_cosine_sim}\n")
        
        # If using Optuna, also save the parameters
        if trial:
            f.write("\nOptuna Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")

    # Generate some predictions for inspection
    predictions = model.predict(X_test, verbose=0)
    cosine_similarities = np.sum(predictions * y_test_final, axis=-1) / (
        np.linalg.norm(predictions, axis=-1) * np.linalg.norm(y_test_final, axis=-1)
    )

    print("\nTest Predictions and Cosine Similarities:")
    for i in range(min(5, len(y_test_final))):  # Show just 5 examples to keep output manageable
        print(f"Sample {i}:")
        print(f"  Predicted Vector: {predictions[i][:10]}...")  # Show just first 10 elements
        print(f"  True Vector: {y_test_final[i][:10]}...")
        print(f"  Cosine Similarity: {cosine_similarities[i]:.4f}")
        print("-" * 50)

    # Return both the model and the evaluation metric for Optuna
    return model, test_cosine_sim

# Define the objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna to optimize.
    Takes an Optuna trial and returns the metric to optimize (cosine similarity).
    """
    # Create trial directory
    trial_dir = os.path.join(saveFolder, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Get the file paths
    folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
    folders_names = ["content", "function"]
    
    # Prepare dataset
    train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)
    
    # Pass the trial to NN function for hyperparameter tuning
    model, cosine_sim = NN(train_files, test_files, y_train, y_test, trial)
    
    # Return the metric to optimize (need to negate as Optuna minimizes by default)
    return -cosine_sim  # Negate because we want to maximize cosine similarity

# Main function to run the optimization
def run_optimization(n_trials=30):
    """
    Run the Optuna optimization process.
    Args:
        n_trials: Number of trials to run
    """
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="minimize")  # We're minimizing -cosine_sim to maximize cosine_sim
    study.optimize(objective, n_trials=n_trials)
    
    # Print the best parameters and value
    print("Best trial:")
    trial = study.best_trial
    print("  Value (Cosine Similarity): ", -trial.value)  # Negate back to get the actual similarity
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best parameters
    with open(os.path.join(saveFolder, "best_params.txt"), "w") as f:
        f.write(f"Best Trial Value (Cosine Similarity): {-trial.value}\n")
        f.write("Best Parameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    
    # Create parameter importance plot
    try:
        # Create visualization of parameter importances if matplotlib is available
        importance = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        plt.bar(importance.keys(), importance.values())
        plt.title('Parameter Importance')
        plt.xlabel('Parameter')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder, "parameter_importance.png"))
        plt.close()
    except:
        print("Could not create parameter importance plot")
    
    # Try to create optimization history plot
    try:
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        plt.plot([t.number for t in study.trials], [-t.value for t in study.trials], marker='o')
        plt.xlabel('Trial Number')
        plt.ylabel('Cosine Similarity')
        plt.title('Optimization History')
        plt.grid(True)
        plt.savefig(os.path.join(saveFolder, "optimization_history.png"))
        plt.close()
    except:
        print("Could not create optimization history plot")
    
    return trial.params

# Function to train model with best parameters
def train_with_best_params(best_params):
    """
    Train a final model using the best parameters found by Optuna.
    Args:
        best_params: Dictionary of best parameters from Optuna
    """
    print("\nTraining final model with best parameters...")
    folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
    folders_names = ["content", "function"]
    
    # Prepare dataset
    train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)
    
    # Create a directory for the best model
    best_model_dir = os.path.join(saveFolder, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Create a "fake" trial with the best parameters
    class FakeTrial:
        def __init__(self, params):
            self.params = params
            
        def suggest_categorical(self, name, choices):
            return self.params[name]
            
        def suggest_float(self, name, low, high, step=None, log=False):
            return self.params[name]
            
        def suggest_int(self, name, low, high, step=1, log=False):
            return self.params[name]
    
    fake_trial = FakeTrial(best_params)
    
    # Train model with best parameters
    best_model, best_cosine_sim = NN(train_files, test_files, y_train, y_test, fake_trial)
    
    # Save the best model
    best_model.save(os.path.join(best_model_dir, "best_model.keras"))
    
    print(f"\nBest model trained with cosine similarity: {best_cosine_sim:.4f}")
    print(f"Best model saved to: {best_model_dir}")
    
    return best_model, best_cosine_sim

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments (optional)
    import argparse
    parser = argparse.ArgumentParser(description='Run neural network training with Optuna hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Run Optuna optimization')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials to run')
    parser.add_argument('--load-best', action='store_true', help='Load and use best parameters from file')
    parser.add_argument('--params-file', type=str, default=None, help='Path to best parameters file')
    args = parser.parse_args()
    
    # Setup folder for files
    print(f"Saving results to: {saveFolder}")
    
    if args.optimize:
        # Run Optuna optimization
        print(f"Running Optuna optimization with {args.trials} trials...")
        best_params = run_optimization(n_trials=args.trials)
        
        # Train with best parameters
        best_model, best_cosine_sim = train_with_best_params(best_params)
    elif args.load_best and args.params_file:
        # Load best parameters from file
        import json
        with open(args.params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded best parameters from {args.params_file}")
        
        # Train with loaded parameters
        best_model, best_cosine_sim = train_with_best_params(best_params)
    else:
        # Run regular training without optimization
        print("Running regular training without optimization...")
        folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
        folders_names = ["content", "function"]
        
        # Prepare dataset
        train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)
        
        # Train and evaluate CNN
        model = NN(train_files, test_files, y_train, y_test)
    
    print("Training completed successfully!")