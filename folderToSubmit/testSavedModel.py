import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return 1 - K.sum(y_true * y_pred, axis=-1)

def cosine_similarity(y_true, y_pred):
    return K.sum(y_true * y_pred, axis=-1) / (K.sqrt(K.sum(y_true**2, axis=-1)) * K.sqrt(K.sum(y_pred**2, axis=-1)))

folder = "models/finalEEG2vec"     ##change this to look at different runs
model_path = os.path.join(folder, "model_epoch01_valloss0.3238.keras")
data_path = os.path.join(folder, "test_data.npz")

if not (os.path.exists(model_path)):
    print("Model path does not exist!")
    exit()

if not (os.path.exists(data_path)):
    print("Data path does not exist!")
    exit()

#load trinaed model
print(f"Loading model from: {model_path}")
model = load_model(model_path, custom_objects={
    "cosine_similarity_loss": cosine_similarity_loss,
    "cosine_similarity": cosine_similarity
})

#load saved test data (make sure it corresponds with this model, should be from the same folder in "savedNLPmodels")
print(f"Loading test data from: {data_path}")
data = np.load(data_path, allow_pickle=True)  

X_test = data["X_test"]
y_test = data["y_test"]

print(f"X_test original shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#reformat the data
#convert from (32, n_samples, 56, 107, 1) to a list of 32 inputs, each with shape (n_samples, 56, 107, 1)
X_test_list = [X_test[i] for i in range(X_test.shape[0])]

print(f"Number of input arrays: {len(X_test_list)}")
print(f"Shape of first input array: {X_test_list[0].shape}")

print("Evaluating model...")
test_loss, test_acc, test_cosine_sim = model.evaluate(X_test_list, y_test)
predictions_test = model.predict(X_test_list, verbose=0)
# np.savez_compressed("test_data.npz", 
#                     predictions_test=predictions_test)

print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Cosine Similarity: {test_cosine_sim}")

#this section runs predictions on x_train and x_valid for use in vec2cf, comment this out if not needed

train_path = os.path.join(folder, "train_data.npz")
valid_path = os.path.join(folder, "valid_data.npz")

train_data = np.load(train_path, allow_pickle=True)  
X_train = train_data["X_train"]

valid_data = np.load(valid_path, allow_pickle=True)  
X_valid = valid_data["X_valid"]

#reformat loaded data
X_train_list = [X_train[i] for i in range(X_train.shape[0])]
X_valid_list = [X_valid[i] for i in range(X_train.shape[0])]

print ("X_train shape:",X_train.shape)
print ("X_valid shape:",X_valid.shape)

predictionsXtrain = model.predict(X_train_list, verbose=0)
predictionsXvalid = model.predict(X_valid_list, verbose=0)

predictions_train = np.concatenate([predictionsXtrain, predictionsXvalid], axis=0)

# np.savez_compressed("train_data.npz", 
#                     predictions_train=predictions_train)