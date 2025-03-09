# For images
from PIL import Image
import os
# For CNN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

tensorflow.config.set_visible_devices([], 'GPU')


####
# Code for the very first version of the CNN architecture for the POS dataset
####

# Function training and testing the CNN
def CNN_for_data(folders_path, folders_names, no_classes, no_cnn_trained, no_batch_size, momentum_value):
        
    training_images, training_labels, testing_images, testing_labels = CNN_data_preparation(folders_path, folders_names, datasets, new_size, no_classes=2, no_spect_in_training = 35)

    for k in range(no_cnn_trained):

        inputs = [Input(shape=(56, 128, 3)) for eeg in training_images]

        # Creating convolution element for each input
        eeg_cnn_elements = []
        for input in inputs:
            x = Conv2D(filters=3, kernel_size=(3,3), activation='relu', padding='same')(input)
            x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
            x = Flatten()(x)
            eeg_cnn_elements.append(x)

        # Concentrate all branches
        combined = Concatenate()(eeg_cnn_elements)

        # Adding more layers
        output = Dense(units=no_classes, activation='softmax')(combined)
        model = Model(inputs=inputs, outputs=output)

        # Learning parameters

        model.compile(optimizer=SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        no_epochs = 90

        history = model.fit(x=training_images, y=training_labels, batch_size=no_batch_size, epochs=no_epochs, verbose=1, shuffle=True)
        test_loss, test_acc = model.evaluate(testing_images, testing_labels)

        print("Test loss: {}, Test Accuracy: {}".format(test_loss,test_acc))

        # Confusion Matrix
        predicted_classes = model.predict(testing_images)
        predicted_classes_cathegories = np.argmax(predicted_classes, axis=1)

        true_labels = np.argmax(testing_labels, axis=1)
        
        confusion_mtx = confusion_matrix(true_labels, predicted_classes_cathegories)

        fig, ax = plt.subplots(figsize=(15,10))
        ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'ConfusionMatrix_{timestamp}.png')
        plt.close()
        print(timestamp )
        

############################################################################################################
# Code for data division into testing and training groups
def CNN_data_preparation(folders_path, folder_names, datasets, new_size, no_classes, no_spect_in_training):
    #array of numpyarrays for images, labels need just array of labels

    training_images = [[] for i in range(32)]
    training_labels = []

    testing_images = [[] for i in range(32)]
    testing_labels = []

    # Looping through 'content' vs 'function'
    for i in range(len(folder_names)):
        current_path = os.path.join(folders_path, folder_names[i])

        # Looping through participants
        for patient_name in os.listdir(current_path):

            # Temporary variables to hold data before dividing into testing and training dataset
            images_hold = [[] for i in range(32)]
            labels_hold = []

            current_patient_path = os.path.join(current_path, patient_name)

            # Looping through the samples in a patient
            for sample_name in os.listdir(current_patient_path):
                current_sample_path = os.path.join(current_patient_path, sample_name)

                indx_image = 0
                labels_hold.append(datasets[i])

                # Looping through images
                for spectrogram_name in os.listdir(current_sample_path):

                    current_image_path = os.path.join(current_sample_path, spectrogram_name)
                    image = Image.open(current_image_path)
                    image = image.resize(new_size)
                    image = np.array(image)

                    images_hold[indx_image].append(image)
                    indx_image = indx_image + 1

            # Dividing images from this patient into training and testing dataset
            indx_eeg = 0

            for eeg_sensor in images_hold:

                training_images[indx_eeg].extend(eeg_sensor[:no_spect_in_training])
                testing_images[indx_eeg].extend(eeg_sensor[no_spect_in_training:])
                
                indx_eeg = indx_eeg + 1
                                

            training_labels.extend(labels_hold[:no_spect_in_training])
            testing_labels.extend(labels_hold[no_spect_in_training:])

    training_images = [np.array(training_eeg_array) for training_eeg_array in training_images]
    testing_images = [np.array(testing_eeg_array) for testing_eeg_array in testing_images]


    training_labels = np.array(keras.utils.to_categorical(training_labels, no_classes))
    testing_labels = np.array(keras.utils.to_categorical(testing_labels, no_classes))


    return training_images, training_labels, testing_images, testing_labels

# Print time to see when code started running
now0 = datetime.now()
time = now0.strftime("%Y%m%d_%H%M%S")
print(time)

# Folders according to where the data is 
current_directory = os.path.dirname(os.path.abspath(__file__))
folders_path = os.path.join(current_directory, "spectrogramDataColour")
folders_names = ["content", "function"]

# Variables that can be changed
datasets = [0, 1]
no_classes = 2
no_cnn_trained = 2
no_batch_size = 5 ##changed from 20 to 5 to try and save memory
momentum_value = 0
new_size = (56,128)


#training_images, training_labels, testing_images, testing_labels = CNN_data_preparation(folders_path, folders_names, datasets, new_size, no_classes=2, no_spect_in_training = 35)
CNN_for_data(folders_path, folders_names, no_classes, no_cnn_trained, no_batch_size, momentum_value)
