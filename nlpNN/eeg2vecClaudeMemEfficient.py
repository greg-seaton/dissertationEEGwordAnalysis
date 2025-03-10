from PIL import Image
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import re
import gc

# Set memory options for TensorFlow
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Load GloVe model
import gensim.downloader as api
print("Loading GloVe model...")
NLPmodel = api.load("glove-wiki-gigaword-100")  # 100D, ~91MB
print("GloVe model loaded.")

def removeIndexAndCheck(word):
    return re.sub(r"\d+$", "", word)

# Removes index from the end of the word and gets its vector
# Alerts user if word is not contained in the model
def getVector(word):
    word = re.sub(r"\d+$", "", word.lower())
    if word in NLPmodel:
        return NLPmodel[word]
    else:
        print(f"Warning: Word '{word}' not found in model vocabulary")
        return np.zeros(100)

def cosine_similarity_loss(y_true, y_pred):
    # Add small epsilon to prevent division by zero
    epsilon = 1e-7
    y_true_norm = K.l2_normalize(y_true, axis=-1) 
    y_pred_norm = K.l2_normalize(y_pred, axis=-1)
    cosine = K.sum(y_true_norm * y_pred_norm, axis=-1)
    return 1 - cosine

def NN_prep(folders_path, folders_names):
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
                    # Limit number of samples per participant if needed
                    # participant_words = participant_words[:100]  # Uncomment to use subset
                    split_idx = int(len(participant_words) * 0.7)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    test_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])
                    y_train.extend(getVector(word) for word in participant_words[:split_idx])
                    y_test.extend(getVector(word) for word in participant_words[split_idx:])
        else:
            print(f"Folder not found: {full_path}")

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    print(f"Length of y_train: {len(y_train)}")
    print(f"Length of y_test: {len(y_test)}")
    return train_files, test_files, y_train, y_test

def load_sample(folder_path):
    """Loads and processes BMP files into 56x107x1 spectrograms."""
    files = sorted(os.listdir(folder_path))[:32]  
    sample_data = np.zeros((32, 56, 107, 1), dtype=np.float32)  

    for i, file in enumerate(files):
        if i >= 32:  # Ensure we only process 32 files
            break
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("L").resize((107, 56))
        sample_data[i, :, :, 0] = np.array(image, dtype=np.float32) / 255.0  # Normalize

    return sample_data

def create_single_spectrogram_model():
    """Creates a simple CNN model for a single spectrogram."""
    input_layer = Input(shape=(56, 107, 1))
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_layer)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(100, activation='linear')(x)
    model = Model(input_layer, output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=cosine_similarity_loss
    )
    
    return model

def train_memory_efficient(train_files, test_files, y_train, y_test, epochs=20):
    """Memory-efficient training approach that processes one spectrogram at a time."""
    
    # Create and compile model
    print("Creating model...")
    K.clear_session()
    gc.collect()
    model = create_single_spectrogram_model()
    model.summary()
    
    # Function to predict word vector by averaging predictions from all spectrograms
    def predict_word(sample_path):
        spectrograms = load_sample(sample_path)
        predictions = []
        for j in range(min(32, spectrograms.shape[0])):
            single_pred = model.predict(spectrograms[j:j+1], verbose=0)
            predictions.append(single_pred[0])
        return np.mean(predictions, axis=0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loss_sum = 0
        batches = 0
        
        # Shuffle training data
        indices = np.arange(len(train_files))
        np.random.shuffle(indices)
        
        # Process each training sample (word)
        for i, idx in enumerate(indices):
            if i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(indices)}")
            
            # Load the spectrograms for this word
            try:
                spectrograms = load_sample(train_files[idx])
                target_vector = y_train[idx]
                
                # Train on each spectrogram individually with the same target
                batch_loss = 0
                for j in range(min(32, spectrograms.shape[0])):
                    single_spec = spectrograms[j:j+1]  # Take just one spectrogram
                    loss = model.train_on_batch(single_spec, np.expand_dims(target_vector, axis=0))
                    batch_loss += loss
                
                # Average loss across the spectrograms
                batch_loss /= min(32, spectrograms.shape[0])
                loss_sum += batch_loss
                batches += 1
            except Exception as e:
                print(f"Error processing {train_files[idx]}: {e}")
                continue
            
            # Clear memory periodically
            if i % 50 == 0:
                gc.collect()
        
        # Report epoch results
        if batches > 0:
            avg_loss = loss_sum / batches
            print(f"  Average loss: {avg_loss:.4f}")
        
        # Evaluate on a small subset of test data after each epoch
        if epoch % 5 == 0 or epoch == epochs-1:
            print("\nEvaluating on test subset:")
            test_subset = min(50, len(test_files))
            cosine_similarities = []
            
            for i in range(test_subset):
                try:
                    pred_vector = predict_word(test_files[i])
                    true_vector = y_test[i]
                    
                    # Calculate cosine similarity
                    norm_pred = np.linalg.norm(pred_vector)
                    norm_true = np.linalg.norm(true_vector)
                    
                    if norm_pred > 0 and norm_true > 0:
                        cosine_sim = np.dot(pred_vector, true_vector) / (norm_pred * norm_true)
                        cosine_similarities.append(cosine_sim)
                except Exception as e:
                    print(f"Error evaluating {test_files[i]}: {e}")
                    continue
            
            if cosine_similarities:
                print(f"Subset average cosine similarity: {np.mean(cosine_similarities):.4f}")
    
    # Final evaluation on the full test set
    print("\nFinal evaluation on test set:")
    cosine_similarities = []
    unique_predictions = set()
    
    for i in range(len(test_files)):
        if i % 10 == 0:
            print(f"  Testing sample {i+1}/{len(test_files)}")
        
        try:
            pred_vector = predict_word(test_files[i])
            true_vector = y_test[i]
            
            # Track unique predictions to check for model collapse
            pred_tuple = tuple(np.round(pred_vector[:5], 4))
            unique_predictions.add(pred_tuple)
            
            # Calculate cosine similarity
            norm_pred = np.linalg.norm(pred_vector)
            norm_true = np.linalg.norm(true_vector)
            
            if norm_pred > 0 and norm_true > 0:
                cosine_sim = np.dot(pred_vector, true_vector) / (norm_pred * norm_true)
                cosine_similarities.append(cosine_sim)
                
                # Print a few examples
                if i < 15:
                    print(f"Sample {i}:")
                    print(f"  Predicted Vector: {pred_vector[:15]}...")
                    print(f"  True Vector: {true_vector[:15]}...")
                    print(f"  Cosine Similarity: {cosine_sim:.4f}")
                    print("-" * 50)
        except Exception as e:
            print(f"Error evaluating {test_files[i]}: {e}")
            continue
    
    if cosine_similarities:
        print(f"Average cosine similarity: {np.mean(cosine_similarities):.4f}")
    print(f"Number of unique prediction patterns: {len(unique_predictions)}")
    
    return model, predict_word

# Main execution
if __name__ == "__main__":
    # Clear memory before starting
    K.clear_session()
    gc.collect()
    
    # Paths
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folders_path = os.path.join(current_directory, "../dataSets/spectrogramDataHighGran")
    folders_names = ["content", "function"]
    
    # Prepare dataset
    train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)
    
    # Use subset of data if needed (uncomment to use)
    # max_samples = 200
    # if len(train_files) > max_samples:
    #     train_files = train_files[:max_samples]
    #     y_train = y_train[:max_samples]
    # if len(test_files) > max_samples//2:
    #     test_files = test_files[:max_samples//2]
    #     y_test = y_test[:max_samples//2]
    
    # Train model with memory-efficient approach
    model, predict_word = train_memory_efficient(
        train_files, 
        test_files, 
        y_train, 
        y_test,
        epochs=20  # Adjust as needed
    )
    
    # Save model if desired
    model.save('../savedModels/spectrogram_word_model.keras')
    
    print("Training complete!")