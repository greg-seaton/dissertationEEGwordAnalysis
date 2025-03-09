import os
import random
import tensorflow as tf
import numpy as np

# Content and function each contain 50 words. This script is initially designed for one subject.
# Each folder for a different word should be split into different test and train sets.

def process_csv_file(file_path):
    """
    Process each CSV file containing 96 integers.
    The model processes the integers and outputs a single value.
    """
    data = np.loadtxt(file_path, delimiter=',')  # Assuming CSV is comma-separated; adjust if needed
    data = tf.convert_to_tensor(data, dtype=tf.float32)  # Convert to TensorFlow tensor

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(96,)),  # Input layer with 96 features (the CSV data)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output a single value for each CSV file
    ])
    
    return model(data)

def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}  # Assign labels

    for i in range(len(folders_names)):  # Loops through content and function
        folder = folders_names[i]
        full_path = os.path.join(folders_path, folder)

        if os.path.exists(full_path):
            print(f"\nContents of {folder}:")
            participants = os.listdir(full_path)

            for participant in participants:  # Loops through participants
                participant_path = os.path.join(full_path, participant)

                if os.path.isdir(participant_path):
                    print("Participant:", participant)
                    participant_words = os.listdir(participant_path)  # Get all word data from participant

                    # Convert to TensorFlow dataset and shuffle
                    dataset = tf.data.Dataset.from_tensor_slices(participant_words)
                    dataset = dataset.shuffle(len(participant_words), seed=42)

                    # Train-test split
                    split_idx = int(len(participant_words) * 0.8)
                    train_files = dataset.take(split_idx)
                    test_files = dataset.skip(split_idx)

                    # Generate labels (0 for content, 1 for function)
                    label = label_map[folder]
                    y_train = [label] * split_idx
                    y_test = [label] * (len(participant_words) - split_idx)

                    print(f"Train: {train_files} (Labels: {y_train})")
                    print(f"Test: {test_files} (Labels: {y_test})")

                    # Process each word's files (each word is a folder with 32 CSV files)
                    def process_word(word):
                        # Convert the tensor to a string
                        word = word.numpy().decode("utf-8")  # Convert tensor to string

                        word_path = os.path.join(participant_path, word)
                        csv_files = [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith('.csv')]

                        word_outputs = [process_csv_file(csv) for csv in csv_files]  # Process each CSV file
                        word_output = tf.concat(word_outputs, axis=-1)  # Concatenate outputs from all CSV files (32 outputs)

                        return word_output


                    # Process training and testing data
                    train_dataset = train_files.map(process_word)
                    test_dataset = test_files.map(process_word)

                    # Zip the data with labels
                    train_dataset = tf.data.Dataset.zip((train_dataset, tf.data.Dataset.from_tensor_slices(y_train)))
                    test_dataset = tf.data.Dataset.zip((test_dataset, tf.data.Dataset.from_tensor_slices(y_test)))

                    # Optionally, batch, cache, and prefetch for optimal performance
                    batch_size = 32
                    train_dataset = train_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
                    test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

                    # Model Architecture: Final Classifier
                    def build_final_model():
                        # Define the final classifier model
                        model_input = tf.keras.Input(shape=(32,))  # 32 features from the 32 CSV files' outputs
                        x = tf.keras.layers.Dense(128, activation='relu')(model_input)
                        x = tf.keras.layers.Dense(64, activation='relu')(x)
                        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification (0 or 1)

                        model = tf.keras.Model(inputs=model_input, outputs=output)
                        return model

                    final_model = build_final_model()

                    # Compile and train the final model
                    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    final_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

        else:  # content/function folder not found
            print(f"Folder not found: {full_path}")

current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataConverted1channel")
folders_names = ["content", "function"]  # content = 0, function = 1

NN_prep(folders_path, folders_names)
