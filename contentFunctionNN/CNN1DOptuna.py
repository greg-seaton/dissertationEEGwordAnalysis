import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
import optuna
from optuna.integration import TFKerasPruningCallback

def NN_prep(folders_path, folders_names):
    label_map = {"content": 0, "function": 1}
    train_files = []
    test_files = []
    y_train = []
    y_test = []

    for folder in folders_names:
        full_path = os.path.join(folders_path, folder)
        if os.path.exists(full_path):
            participants = os.listdir(full_path)
            for participant in participants:
                participant_path = os.path.join(full_path, participant)
                if os.path.isdir(participant_path):
                    participant_words = os.listdir(participant_path)
                    random.shuffle(participant_words)
                    participant_words = participant_words[:800]
                    split_idx = int(len(participant_words) * 0.8)
                    train_files.extend([os.path.join(participant_path, word) for word in participant_words[:split_idx]])
                    test_files.extend([os.path.join(participant_path, word) for word in participant_words[split_idx:]])
                    label = label_map[folder]
                    y_train.extend([label] * split_idx)
                    y_test.extend([label] * (len(participant_words) - split_idx))
        else:
            print(f"Folder not found: {full_path}")

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    return train_files, test_files, y_train, y_test

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
    filters = trial.suggest_int('filters', 8, 64, step=8)
    kernel_size = trial.suggest_int('kernel_size', 3, 9, step=2)
    pool_size = trial.suggest_int('pool_size', 2, 4)
    dense_units = trial.suggest_int('dense_units', 128, 512, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    inputs = [Input(shape=(96, 1)) for _ in range(32)]
    cnn_outputs = []
    for inp in inputs:
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(inp)
        x = MaxPool1D(pool_size)(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    combined = Concatenate()(cnn_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    
    if optimizer_type == 'sgd':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial, X_train, X_val, y_train, y_val):
    model = create_model(trial)
    batch_size = trial.params['batch_size']
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )
    return np.max(history.history['val_accuracy'])

# Main execution
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataGoated")
folders_names = ["content", "function"]

train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

# Load and preprocess data
X_train_full_samples = [load_sample(file) for file in train_files]
X_test_samples = [load_sample(file) for file in test_files]

X_train_full = [np.array([x[i] for x in X_train_full_samples])[:, :, np.newaxis] for i in range(32)]
X_test = [np.array([x[i] for x in X_test_samples])[:, :, np.newaxis] for i in range(32)]
y_train = np.array(y_train)
y_test = np.array(y_test)

# Split into training and validation
n_samples = y_train.shape[0]
indices = np.random.permutation(n_samples)
split_idx = int(0.8 * n_samples)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train = [x[train_idx] for x in X_train_full]
X_val = [x[val_idx] for x in X_train_full]
y_train_new, y_val = y_train[train_idx], y_train[val_idx]

# Optuna study
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(lambda trial: objective(trial, X_train, X_val, y_train_new, y_val), n_trials=50)

# Retrain best model on full data
best_params = study.best_params
model = create_model(study.best_trial)
model.fit(X_train_full, y_train, epochs=100, batch_size=best_params['batch_size'], verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")


# (base) greg@Greg-laptop:~/Documents/GregCode$ python CNNcurrentOptuna.py 
# 2025-02-13 18:30:03.271096: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1739471403.309644    4900 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# E0000 00:00:1739471403.318250    4900 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2025-02-13 18:30:03.357008: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# [I 2025-02-13 18:31:41,158] A new study created in memory with name: no-name-d832807b-63d3-430f-bc30-9b32bf461ad4
# 2025-02-13 18:31:41.211563: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
# [I 2025-02-13 18:33:44,346] Trial 0 finished with value: 0.53515625 and parameters: {'filters': 16, 'kernel_size': 5, 'pool_size': 3, 'dense_units': 384, 'dropout_rate': 0.457748899498666, 'optimizer': 'adam', 'learning_rate': 0.004778845239563838, 'batch_size': 256}. Best is trial 0 with value: 0.53515625.
# [I 2025-02-13 18:37:39,168] Trial 1 finished with value: 0.5546875 and parameters: {'filters': 56, 'kernel_size': 9, 'pool_size': 4, 'dense_units': 128, 'dropout_rate': 0.4639113264139353, 'optimizer': 'adam', 'learning_rate': 0.008935917449691504, 'batch_size': 64}. Best is trial 1 with value: 0.5546875.
# [I 2025-02-13 18:45:07,889] Trial 2 finished with value: 0.51953125 and parameters: {'filters': 64, 'kernel_size': 5, 'pool_size': 2, 'dense_units': 384, 'dropout_rate': 0.3805165170645499, 'optimizer': 'adam', 'learning_rate': 0.00015988089668129846, 'batch_size': 256}. Best is trial 1 with value: 0.5546875.
# [I 2025-02-13 18:45:52,458] Trial 3 finished with value: 0.51953125 and parameters: {'filters': 8, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.45768977333578265, 'optimizer': 'adam', 'learning_rate': 0.0006363788022430667, 'batch_size': 128}. Best is trial 1 with value: 0.5546875.
# [I 2025-02-13 18:47:10,199] Trial 4 finished with value: 0.51953125 and parameters: {'filters': 16, 'kernel_size': 7, 'pool_size': 2, 'dense_units': 256, 'dropout_rate': 0.3549243396870177, 'optimizer': 'adam', 'learning_rate': 0.004004453548296268, 'batch_size': 256}. Best is trial 1 with value: 0.5546875.
# [I 2025-02-13 19:05:43,225] Trial 5 finished with value: 0.5234375 and parameters: {'filters': 56, 'kernel_size': 3, 'pool_size': 4, 'dense_units': 128, 'dropout_rate': 0.43871240206590834, 'optimizer': 'adam', 'learning_rate': 0.006909414200976004, 'batch_size': 128}. Best is trial 1 with value: 0.5546875.
# [I 2025-02-13 19:08:13,146] Trial 6 finished with value: 0.55859375 and parameters: {'filters': 48, 'kernel_size': 7, 'pool_size': 4, 'dense_units': 384, 'dropout_rate': 0.34205420868281966, 'optimizer': 'sgd', 'learning_rate': 0.003856561139474552, 'batch_size': 128, 'momentum': 0.822542117825407}. Best is trial 6 with value: 0.55859375.
# [I 2025-02-13 19:08:21,910] Trial 7 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:08:28,733] Trial 8 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:08:37,663] Trial 9 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:08:43,114] Trial 10 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:08:48,963] Trial 11 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:08:54,961] Trial 12 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:09:00,689] Trial 13 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:09:08,400] Trial 14 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:14:43,214] Trial 15 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:18:40,194] Trial 16 finished with value: 0.5390625 and parameters: {'filters': 56, 'kernel_size': 7, 'pool_size': 3, 'dense_units': 320, 'dropout_rate': 0.4913018214922743, 'optimizer': 'sgd', 'learning_rate': 0.00024787944749587705, 'batch_size': 64, 'momentum': 0.852484345230457}. Best is trial 6 with value: 0.55859375.
# [I 2025-02-13 19:18:45,404] Trial 17 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:18:51,746] Trial 18 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 19:19:00,089] Trial 19 pruned. Trial was pruned at epoch 0.
# WARNING:tensorflow:5 out of the last 16 calls to <function TensorFlowTrainer._make_function.<locals>.multi_step_on_iterator at 0x7f4693c86d40> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
# [I 2025-02-13 19:21:43,503] Trial 20 finished with value: 0.5703125 and parameters: {'filters': 40, 'kernel_size': 9, 'pool_size': 4, 'dense_units': 320, 'dropout_rate': 0.2610620925385121, 'optimizer': 'adam', 'learning_rate': 0.005787414418461204, 'batch_size': 128}. Best is trial 20 with value: 0.5703125.
# [I 2025-02-13 20:36:49,558] Trial 21 finished with value: 0.52734375 and parameters: {'filters': 40, 'kernel_size': 9, 'pool_size': 4, 'dense_units': 320, 'dropout_rate': 0.2560245781930528, 'optimizer': 'adam', 'learning_rate': 0.0058600003190267536, 'batch_size': 128}. Best is trial 20 with value: 0.5703125.
# [I 2025-02-13 20:36:59,192] Trial 22 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 20:37:06,778] Trial 23 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 20:52:22,883] Trial 24 finished with value: 0.5546875 and parameters: {'filters': 40, 'kernel_size': 9, 'pool_size': 3, 'dense_units': 256, 'dropout_rate': 0.3318158674041396, 'optimizer': 'adam', 'learning_rate': 0.009137152038433865, 'batch_size': 128}. Best is trial 20 with value: 0.5703125.
# [I 2025-02-13 20:52:41,616] Trial 25 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 20:52:57,096] Trial 26 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 20:53:13,448] Trial 27 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 20:56:35,304] Trial 28 finished with value: 0.5390625 and parameters: {'filters': 48, 'kernel_size': 7, 'pool_size': 3, 'dense_units': 384, 'dropout_rate': 0.41437102813164783, 'optimizer': 'sgd', 'learning_rate': 0.0039897537051194, 'batch_size': 256, 'momentum': 0.8876608099001887}. Best is trial 20 with value: 0.5703125.
# [I 2025-02-13 21:00:20,238] Trial 29 finished with value: 0.5234375 and parameters: {'filters': 24, 'kernel_size': 5, 'pool_size': 3, 'dense_units': 320, 'dropout_rate': 0.32438961911588526, 'optimizer': 'adam', 'learning_rate': 0.0047295024264925175, 'batch_size': 64}. Best is trial 20 with value: 0.5703125.
# [I 2025-02-13 21:00:39,723] Trial 30 pruned. Trial was pruned at epoch 0.
# Killed