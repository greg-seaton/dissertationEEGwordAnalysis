import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

# Define custom loss function
def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

# Set folder path where model and test data are saved
folder = "2025-03-12_10-43"     ##change this to look at different runs
model_path = os.path.join("savedNLPmodels", folder, "model_epoch02_valloss0.9231.keras")
data_path = os.path.join("savedNLPmodels", folder, "test_data.npz")

            ##if it says file not found, it probably got corrupted if the program crashed while it was saving

if not (os.path.exists(model_path)):
    print ("path does not exist!")

# Load the trained model
print(f"Loading model from: {model_path}")
model = load_model(model_path, custom_objects={"cosine_similarity_loss": cosine_similarity_loss})

# Load test dataset
print(f"Loading test data from: {data_path}")
data = np.load(data_path, allow_pickle=True)  

X_test = data["X_test"]
y_test = data["y_test"]

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

X_test = np.transpose(X_test, (1, 0, 2, 3, 4))  # Swaps first and second axes
X_test_split = [X_test[:, i] for i in range(32)]  

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Evaluate model
print("Evaluating model...")
test_loss, test_acc, test_cosine_sim = model.evaluate(X_test_split, y_test)  # Pass as a list

print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")
