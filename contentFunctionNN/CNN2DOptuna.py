import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Concatenate, Reshape
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
    """Loads and processes 32 CSV files into 12x8 spectrograms"""
    sample_data = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, header=None).values.flatten()
        # Reshape to 12x8 and add channel dimension
        data = data.reshape((12, 8, 1)).astype(np.float32)
        sample_data.append(data)
    if len(sample_data) != 32:
        print(f"Warning: {folder_path} has {len(sample_data)} files! Should be 32")
    return sample_data

def create_model(trial):
    # Hyperparameter search space
    filters = trial.suggest_int('filters', 8, 64, step=8)
    kernel_w = trial.suggest_int('kernel_w', 3, 7, step=2)
    kernel_h = trial.suggest_int('kernel_h', 3, 5, step=2)
    pool_size = trial.suggest_categorical('pool_size', [2, 3])
    dense_units = trial.suggest_int('dense_units', 128, 512, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Input processing
    inputs = [Input(shape=(12, 8, 1)) for _ in range(32)]
    
    cnn_outputs = []
    for inp in inputs:
        x = Conv2D(filters, (kernel_w, kernel_h), activation='relu', padding='same')(inp)
        x = MaxPool2D((pool_size, pool_size))(x)
        x = Flatten()(x)
        cnn_outputs.append(x)
    
    combined = Concatenate()(cnn_outputs)
    x = Dense(dense_units, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Optimizer configuration
    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
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
        epochs=50,
        batch_size=batch_size,
        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
        verbose=0
    )
    return np.max(history.history['val_accuracy'])

def main():
    # Prepare dataset
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folders_path = os.path.join(current_directory, "spectrogramDataGoated")
    folders_names = ["content", "function"]
    train_files, test_files, y_train, y_test = NN_prep(folders_path, folders_names)

    # Load and preprocess data
    X_train_samples = [load_sample(file) for file in train_files]
    X_test_samples = [load_sample(file) for file in test_files]

    # Prepare input arrays (32 inputs each with shape (12, 8, 1))
    X_train = [np.array([sample[i] for sample in X_train_samples]) for i in range(32)]
    X_test = [np.array([sample[i] for sample in X_test_samples]) for i in range(32)]

    # Split into training and validation
    n_samples = len(X_train_samples)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_split = [x[train_idx] for x in X_train]
    X_val = [x[val_idx] for x in X_train]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]

    # Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train_split, X_val, y_train_split, y_val), n_trials=30)

    # Train final model
    best_model = create_model(study.best_trial)
    best_model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=study.best_params['batch_size'],
                  verbose=1)
    
    best_trial = study.best_trial
    best_params = best_trial.params  # Extract best hyperparameters
    print (best_params)
    test_loss, test_acc = best_model.evaluate(X_test, y_test)


    # Train a new model with these parameters
    model = create_model(**best_params)
    model.fit(X_train, y_train, epochs=50, batch_size=256)


    #run this again, i added code to try and run it again with the best params

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()


# (base) greg@Greg-laptop:~/Documents/GregCode$ python CNN2DOptuna.py 
# 2025-02-13 22:48:45.779126: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1739486925.821562  176814 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# E0000 00:00:1739486925.838909  176814 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2025-02-13 22:48:45.892107: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# [I 2025-02-13 22:49:48,183] A new study created in memory with name: no-name-df66bb14-84dc-463d-8a0e-e4a2b1c161c3
# 2025-02-13 22:49:48.226480: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
# [I 2025-02-13 22:51:21,894] Trial 0 finished with value: 0.5390625 and parameters: {'filters': 48, 'kernel_w': 7, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 448, 'dropout_rate': 0.3786653540193652, 'learning_rate': 0.003768337834109027, 'optimizer': 'sgd', 'batch_size': 256}. Best is trial 0 with value: 0.5390625.
# [I 2025-02-13 22:52:21,025] Trial 1 finished with value: 0.55078125 and parameters: {'filters': 32, 'kernel_w': 5, 'kernel_h': 5, 'pool_size': 2, 'dense_units': 256, 'dropout_rate': 0.24531383231414447, 'learning_rate': 0.0030731459952277615, 'optimizer': 'adam', 'batch_size': 256}. Best is trial 1 with value: 0.55078125.
# [I 2025-02-13 22:52:58,207] Trial 2 finished with value: 0.55859375 and parameters: {'filters': 24, 'kernel_w': 7, 'kernel_h': 5, 'pool_size': 3, 'dense_units': 192, 'dropout_rate': 0.33526264909940195, 'learning_rate': 0.00266879645321122, 'optimizer': 'sgd', 'batch_size': 64}. Best is trial 2 with value: 0.55859375.
# [I 2025-02-13 22:53:25,920] Trial 3 finished with value: 0.57421875 and parameters: {'filters': 8, 'kernel_w': 3, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 256, 'dropout_rate': 0.37559630683421585, 'learning_rate': 0.00011818242450863324, 'optimizer': 'adam', 'batch_size': 128}. Best is trial 3 with value: 0.57421875.
# [I 2025-02-13 22:53:52,554] Trial 4 finished with value: 0.57421875 and parameters: {'filters': 8, 'kernel_w': 3, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 512, 'dropout_rate': 0.4228141633564829, 'learning_rate': 0.003681746152661005, 'optimizer': 'sgd', 'batch_size': 128}. Best is trial 3 with value: 0.57421875.
# [I 2025-02-13 22:53:57,149] Trial 5 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 22:54:52,441] Trial 6 finished with value: 0.57421875 and parameters: {'filters': 64, 'kernel_w': 7, 'kernel_h': 3, 'pool_size': 3, 'dense_units': 320, 'dropout_rate': 0.3890364962543077, 'learning_rate': 0.009909827192345592, 'optimizer': 'sgd', 'batch_size': 256}. Best is trial 3 with value: 0.57421875.
# [I 2025-02-13 22:55:00,765] Trial 7 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 22:55:41,609] Trial 8 finished with value: 0.59765625 and parameters: {'filters': 40, 'kernel_w': 7, 'kernel_h': 3, 'pool_size': 3, 'dense_units': 448, 'dropout_rate': 0.318719791116454, 'learning_rate': 0.0018387607023296413, 'optimizer': 'sgd', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 22:56:40,304] Trial 9 finished with value: 0.5390625 and parameters: {'filters': 56, 'kernel_w': 3, 'kernel_h': 3, 'pool_size': 3, 'dense_units': 384, 'dropout_rate': 0.4104407100098758, 'learning_rate': 0.00035503411238648527, 'optimizer': 'adam', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 22:57:36,372] Trial 10 finished with value: 0.5390625 and parameters: {'filters': 40, 'kernel_w': 5, 'kernel_h': 5, 'pool_size': 3, 'dense_units': 512, 'dropout_rate': 0.28792305797022316, 'learning_rate': 0.0012048192743617885, 'optimizer': 'sgd', 'batch_size': 64}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 22:57:43,494] Trial 11 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 22:58:05,253] Trial 12 finished with value: 0.57421875 and parameters: {'filters': 8, 'kernel_w': 3, 'kernel_h': 3, 'pool_size': 3, 'dense_units': 320, 'dropout_rate': 0.2062717375261433, 'learning_rate': 0.00047796991311556294, 'optimizer': 'adam', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 22:59:09,502] Trial 13 finished with value: 0.5546875 and parameters: {'filters': 24, 'kernel_w': 5, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 384, 'dropout_rate': 0.3471699763373771, 'learning_rate': 0.00016168114994153607, 'optimizer': 'adam', 'batch_size': 128}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 22:59:14,122] Trial 14 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 22:59:54,280] Trial 15 finished with value: 0.5390625 and parameters: {'filters': 16, 'kernel_w': 3, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 448, 'dropout_rate': 0.3031222577790957, 'learning_rate': 0.0005801821867015747, 'optimizer': 'adam', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:00:50,419] Trial 16 finished with value: 0.5390625 and parameters: {'filters': 48, 'kernel_w': 7, 'kernel_h': 3, 'pool_size': 3, 'dense_units': 256, 'dropout_rate': 0.3593162212038277, 'learning_rate': 0.00023313683674285957, 'optimizer': 'adam', 'batch_size': 128}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:00:55,046] Trial 17 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 23:01:05,853] Trial 18 pruned. Trial was pruned at epoch 0.
# WARNING:tensorflow:5 out of the last 16 calls to <function TensorFlowTrainer._make_function.<locals>.multi_step_on_iterator at 0x7fae347b0fe0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
# [I 2025-02-13 23:01:11,756] Trial 19 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 23:02:02,599] Trial 20 finished with value: 0.578125 and parameters: {'filters': 32, 'kernel_w': 7, 'kernel_h': 5, 'pool_size': 2, 'dense_units': 320, 'dropout_rate': 0.31597516455614694, 'learning_rate': 0.00034302387602891816, 'optimizer': 'sgd', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:02:07,660] Trial 21 pruned. Trial was pruned at epoch 0.
# WARNING:tensorflow:5 out of the last 256 calls to <function TensorFlowTrainer._make_function.<locals>.multi_step_on_iterator at 0x7fae08ad7920> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
# [I 2025-02-13 23:02:12,382] Trial 22 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 23:02:17,682] Trial 23 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 23:02:21,819] Trial 24 pruned. Trial was pruned at epoch 0.
# [I 2025-02-13 23:03:03,944] Trial 25 finished with value: 0.5390625 and parameters: {'filters': 24, 'kernel_w': 5, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 256, 'dropout_rate': 0.4032127146592428, 'learning_rate': 0.0008240706345053478, 'optimizer': 'sgd', 'batch_size': 128}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:04:10,625] Trial 26 finished with value: 0.546875 and parameters: {'filters': 56, 'kernel_w': 7, 'kernel_h': 5, 'pool_size': 3, 'dense_units': 448, 'dropout_rate': 0.3017125999888778, 'learning_rate': 0.00015286090269810412, 'optimizer': 'adam', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:04:58,282] Trial 27 finished with value: 0.56640625 and parameters: {'filters': 32, 'kernel_w': 5, 'kernel_h': 3, 'pool_size': 2, 'dense_units': 320, 'dropout_rate': 0.3625041692335408, 'learning_rate': 0.00032680334842511447, 'optimizer': 'sgd', 'batch_size': 256}. Best is trial 8 with value: 0.59765625.
# [I 2025-02-13 23:06:00,034] Trial 28 finished with value: 0.6015625 and parameters: {'filters': 40, 'kernel_w': 7, 'kernel_h': 5, 'pool_size': 3, 'dense_units': 192, 'dropout_rate': 0.26281170838881607, 'learning_rate': 0.00200320429637648, 'optimizer': 'adam', 'batch_size': 64}. Best is trial 28 with value: 0.6015625.
# [I 2025-02-13 23:06:05,042] Trial 29 pruned. Trial was pruned at epoch 0.
# Epoch 1/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 8s 60ms/step - accuracy: 0.5133 - loss: 77.6369 
# Epoch 2/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.5051 - loss: 0.7970
# Epoch 3/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 60ms/step - accuracy: 0.5268 - loss: 0.6985
# Epoch 4/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.5231 - loss: 0.6938
# Epoch 5/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 61ms/step - accuracy: 0.4934 - loss: 0.6936
# Epoch 6/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5096 - loss: 0.6922
# Epoch 7/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.5293 - loss: 0.6926
# Epoch 8/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5322 - loss: 0.6923
# Epoch 9/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5274 - loss: 0.6898
# Epoch 10/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5176 - loss: 0.6923
# Epoch 11/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5488 - loss: 0.6872
# Epoch 12/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5529 - loss: 0.6861
# Epoch 13/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5521 - loss: 0.6830
# Epoch 14/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5549 - loss: 0.6862
# Epoch 15/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.5798 - loss: 0.6754 
# Epoch 16/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.6058 - loss: 0.6685
# Epoch 17/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 72ms/step - accuracy: 0.6090 - loss: 0.6659 
# Epoch 18/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 73ms/step - accuracy: 0.6265 - loss: 0.6571 
# Epoch 19/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 71ms/step - accuracy: 0.6096 - loss: 0.6490 
# Epoch 20/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 71ms/step - accuracy: 0.6500 - loss: 0.6318 
# Epoch 21/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 69ms/step - accuracy: 0.6616 - loss: 0.6217 
# Epoch 22/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 73ms/step - accuracy: 0.6984 - loss: 0.5970 
# Epoch 23/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.7186 - loss: 0.5712 
# Epoch 24/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.7562 - loss: 0.5344 
# Epoch 25/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.7083 - loss: 0.5519
# Epoch 26/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.7322 - loss: 0.5401
# Epoch 27/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 60ms/step - accuracy: 0.7773 - loss: 0.4798
# Epoch 28/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 65ms/step - accuracy: 0.8108 - loss: 0.4315
# Epoch 29/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 61ms/step - accuracy: 0.8055 - loss: 0.4235
# Epoch 30/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.8331 - loss: 0.3968
# Epoch 31/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.8421 - loss: 0.3895 
# Epoch 32/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.8536 - loss: 0.3440 
# Epoch 33/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.8820 - loss: 0.3120
# Epoch 34/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 53ms/step - accuracy: 0.8646 - loss: 0.3245
# Epoch 35/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 53ms/step - accuracy: 0.8795 - loss: 0.2842
# Epoch 36/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.9112 - loss: 0.2493
# Epoch 37/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9006 - loss: 0.2367
# Epoch 38/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9337 - loss: 0.1866
# Epoch 39/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9283 - loss: 0.2005
# Epoch 40/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.9398 - loss: 0.1694
# Epoch 41/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 56ms/step - accuracy: 0.8861 - loss: 0.2608
# Epoch 42/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9497 - loss: 0.1525
# Epoch 43/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9550 - loss: 0.1346
# Epoch 44/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.9549 - loss: 0.1294
# Epoch 45/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.9600 - loss: 0.1122 
# Epoch 46/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.8785 - loss: 0.2844
# Epoch 47/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 74ms/step - accuracy: 0.9482 - loss: 0.1489
# Epoch 48/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 67ms/step - accuracy: 0.9621 - loss: 0.1262 
# Epoch 49/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.8675 - loss: 0.3055
# Epoch 50/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9333 - loss: 0.1646
# Epoch 51/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9591 - loss: 0.1142 
# Epoch 52/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9622 - loss: 0.1077
# Epoch 53/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9809 - loss: 0.0799
# Epoch 54/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9760 - loss: 0.0805
# Epoch 55/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9803 - loss: 0.0517
# Epoch 56/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 65ms/step - accuracy: 0.9876 - loss: 0.0489 
# Epoch 57/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9813 - loss: 0.0571
# Epoch 58/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9889 - loss: 0.0431
# Epoch 59/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9942 - loss: 0.0359
# Epoch 60/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9938 - loss: 0.0315 
# Epoch 61/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.9919 - loss: 0.0345 
# Epoch 62/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9930 - loss: 0.0317
# Epoch 63/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9934 - loss: 0.0304
# Epoch 64/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9921 - loss: 0.0309
# Epoch 65/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9918 - loss: 0.0354
# Epoch 66/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9887 - loss: 0.0496
# Epoch 67/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9676 - loss: 0.0863 
# Epoch 68/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9127 - loss: 0.2298
# Epoch 69/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.9708 - loss: 0.0892
# Epoch 70/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 73ms/step - accuracy: 0.9774 - loss: 0.0726
# Epoch 71/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 66ms/step - accuracy: 0.9671 - loss: 0.0886 
# Epoch 72/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 61ms/step - accuracy: 0.9751 - loss: 0.0594
# Epoch 73/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 60ms/step - accuracy: 0.9589 - loss: 0.0960
# Epoch 74/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.9768 - loss: 0.0620
# Epoch 75/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 59ms/step - accuracy: 0.9859 - loss: 0.0525
# Epoch 76/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 60ms/step - accuracy: 0.9903 - loss: 0.0405 
# Epoch 77/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 61ms/step - accuracy: 0.9873 - loss: 0.0353 
# Epoch 78/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9843 - loss: 0.0369
# Epoch 79/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 71ms/step - accuracy: 0.9822 - loss: 0.0411 
# Epoch 80/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 68ms/step - accuracy: 0.9848 - loss: 0.0427
# Epoch 81/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 65ms/step - accuracy: 0.9866 - loss: 0.0295 
# Epoch 82/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9831 - loss: 0.0288
# Epoch 83/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9918 - loss: 0.0253
# Epoch 84/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9903 - loss: 0.0192
# Epoch 85/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9933 - loss: 0.0164
# Epoch 86/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9909 - loss: 0.0223 
# Epoch 87/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9898 - loss: 0.0188
# Epoch 88/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9956 - loss: 0.0159
# Epoch 89/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9904 - loss: 0.0196 
# Epoch 90/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9888 - loss: 0.0208
# Epoch 91/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 66ms/step - accuracy: 0.9874 - loss: 0.0194
# Epoch 92/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.9922 - loss: 0.0142 
# Epoch 93/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9933 - loss: 0.0151
# Epoch 94/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9884 - loss: 0.0185
# Epoch 95/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.9924 - loss: 0.0138 
# Epoch 96/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9875 - loss: 0.0193
# Epoch 97/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9970 - loss: 0.0122
# Epoch 98/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9967 - loss: 0.0183 
# Epoch 99/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 62ms/step - accuracy: 0.9960 - loss: 0.0153
# Epoch 100/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9975 - loss: 0.0141
# 10/10 ━━━━━━━━━━━━━━━━━━━━ 1s 13ms/step - accuracy: 0.4707 - loss: 3.2988 

# Final Test Accuracy: 0.4844