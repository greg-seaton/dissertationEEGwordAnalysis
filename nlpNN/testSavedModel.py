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
folder = "2025-03-12_10-08"     ##change this to look at different runs
model_path = os.path.join("savedNLPmodels", folder, "best_model.keras")     #change this to select the correct model iteration
data_path = os.path.join("savedNLPmodels", folder, "test_data.npz")

print("Checking path:", model_path)
print("Exists:", os.path.exists(model_path))
print("Is a file:", os.path.isfile(model_path))

if not (os.path.exists(model_path)):
    print ("path does not exist!")

# Load the trained model
print(f"Loading model from: {model_path}")
model = load_model(model_path, custom_objects={"cosine_similarity_loss": cosine_similarity_loss})


# Load test dataset
print(f"Loading test data from: {data_path}")
data = np.load(data_path, allow_pickle=True)  # Ensure allow_pickle in case of saved lists

X_test = data["X_test"]
y_test = data["y_test"]

# Convert X_test to NumPy array if it was loaded as a list
X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate model
# Split X_test into 32 separate input arrays
X_test_split = [X_test[:, i] for i in range(32)]  # Create a list of 32 arrays

# Evaluate model
print("Evaluating model...")
test_loss, test_acc, test_cosine_sim = model.evaluate(X_test_split, y_test)  # Pass as a list

print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")
