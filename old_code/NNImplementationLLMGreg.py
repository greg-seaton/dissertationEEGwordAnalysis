#this code is entirely LLM generated, does not currently work (hugely overfits)

import os
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

def CNN_for_data(folders_path, folders_names, no_classes, no_cnn_trained, no_batch_size, momentum_value):
    training_images, training_labels, testing_images, testing_labels = CNN_data_preparation(
        folders_path, folders_names, datasets, no_classes, no_spect_in_training=35
    )

    for k in range(no_cnn_trained):
        # Create 32 input layers, each expecting a 1D vector of 96 elements
        inputs = [Input(shape=(96,)) for _ in range(32)]

        # Process each input with a Dense layer
        dense_outputs = []
        for inp in inputs:
            x = Dense(64, activation='relu')(inp)  # Adjust units as needed
            dense_outputs.append(x)

        # Concatenate all branches
        combined = Concatenate()(dense_outputs)

        # Output layer
        output = Dense(units=no_classes, activation='softmax')(combined)
        model = Model(inputs=inputs, outputs=output)

        # Compile the model
        model.compile(
            optimizer=SGD(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        no_epochs = 90
        history = model.fit(
            x=training_images,
            y=training_labels,
            batch_size=no_batch_size,
            epochs=no_epochs,
            verbose=1,
            shuffle=True
        )
        test_loss, test_acc = model.evaluate(testing_images, testing_labels)
        print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}")


        #extra info
        # Confusion Matrix
        # predicted_classes = model.predict(testing_images)
        # predicted_classes_cathegories = np.argmax(predicted_classes, axis=1)
        # true_labels = np.argmax(testing_labels, axis=1)
        
        # confusion_mtx = confusion_matrix(true_labels, predicted_classes_cathegories)
        # fig, ax = plt.subplots(figsize=(15,10))
        # sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax)
        # ax.set_xlabel("Predicted")
        # ax.set_ylabel("True")
        # ax.set_title("Confusion Matrix")
        
        # now = datetime.now()
        # timestamp = now.strftime("%Y%m%d_%H%M%S")
        # plt.savefig(f'ConfusionMatrix_{timestamp}.png')
        # plt.close()
        # print(timestamp)

def CNN_data_preparation(folders_path, folder_names, datasets, no_classes, no_spect_in_training):
    training_images = [[] for _ in range(32)]
    training_labels = []
    testing_images = [[] for _ in range(32)]
    testing_labels = []

    for i in range(len(folder_names)):
        current_path = os.path.join(folders_path, folder_names[i])
        for patient_name in os.listdir(current_path):
            print (patient_name)###############
            current_patient_path = os.path.join(current_path, patient_name)
            images_hold = [[] for _ in range(32)]
            labels_hold = []

            for sample_name in os.listdir(current_patient_path):
                current_sample_path = os.path.join(current_patient_path, sample_name)

                indx_image = 0
                labels_hold.append(datasets[i])

                # Looping through images
                for spectrogram_name in os.listdir(current_sample_path):

                    csv_path = os.path.join(current_sample_path, spectrogram_name)
                    data = np.loadtxt(csv_path, delimiter=',', dtype=np.int32)
                    images_hold[indx_image].append(data)
                    indx_image = indx_image + 1



            # Split data into training and testing
            for sensor_idx in range(32):
                training_images[sensor_idx].extend(images_hold[sensor_idx][:no_spect_in_training])
                testing_images[sensor_idx].extend(images_hold[sensor_idx][no_spect_in_training:])

            training_labels.extend([datasets[i]] * no_spect_in_training)
            testing_labels.extend([datasets[i]] * (len(images_hold[0]) - no_spect_in_training))

    # Convert to numpy arrays
    training_images = [np.array(sensor) for sensor in training_images]
    testing_images = [np.array(sensor) for sensor in testing_images]

    # Convert labels to categorical
    training_labels = keras.utils.to_categorical(training_labels, no_classes)
    testing_labels = keras.utils.to_categorical(testing_labels, no_classes)

    return training_images, training_labels, testing_images, testing_labels

# Main execution
now0 = datetime.now()
print("Start time:", now0.strftime("%Y%m%d_%H%M%S"))

current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataConverted1channel")
folders_names = ["content", "function"]

datasets = [0, 1]
no_classes = 2
no_cnn_trained = 2
no_batch_size = 20
momentum_value = 0

CNN_for_data(folders_path, folders_names, no_classes, no_cnn_trained, no_batch_size, momentum_value)