import os
import re
import random
import datetime

import cv2
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers import Dense, Dropout
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from nn_register import save_nn_model, plot_model_loss, plot_model_accuracy
from nn_register import model_info_to_txt, plot_predictions

# Global variables
data_folder = './dataset/'
img_width = 160
img_height = 120
filename_pattern = r'(?P<depth_img_number>\d{5})_(?P<control>(left)?(forward)?(right)?).png'
output_pattern = {
    'left': [1, 0, 0], 
    'forward': [0, 1, 0], 
    'right': [0, 0, 1]
}
output_labels = [
    'Left', 
    'Straightforward', 
    'Right'
]
batch_size = 20
epochs = 20
es_patience = 5

def process_output_control(file_names):
    regex = re.compile(filename_pattern)
    control = [regex.search(f).group('control') for f in file_names]
    output_data = [output_pattern.get(c) for c in control]
    return np.array(output_data)

def process_depth_images(file_names):
    data = [cv2.resize(cv2.imread(data_folder + f, \
            cv2.IMREAD_GRAYSCALE), (img_width, img_height)).flatten() \
            for f in file_names]
    return np.array(data, dtype = 'float') / 255.0

def define_model():
    model = Sequential([
        Conv2D(32, (5,5), padding = 'same', input_shape = (img_height,\
                 img_width, 1), activation = 'relu'),
        MaxPooling2D(pool_size = (2,2), strides = 2),
        Conv2D(32, (5,5), padding = 'same', activation = 'relu'),
        AveragePooling2D(pool_size = (2,2), strides = 2),
    	Flatten(),
	    Dense(256, activation = 'relu'),
        Dropout(0.3),
        Dense(128, activation = 'relu'),
	    Dense(3, activation = 'sigmoid')
    ])
    return model


def main():
    # Load dataset and make random sequence
    images = [f for f in os.listdir(data_folder) \
                if os.path.isfile(os.path.join(data_folder, f))]
    random.shuffle(images)

    # Process input and output data
    output_data = process_output_control(images)
    input_data = process_depth_images(images)

    # Split data to training, validation and testing sets (70%, 15%, 15%)
    x_train, x_val_test, y_train, y_val_test = train_test_split(\
                input_data, output_data, test_size = 0.3)
    x_val, x_test, y_val, y_test = train_test_split(\
                x_val_test, y_val_test, test_size = 0.5)

    # Reshape images to CNN input
    x_train = x_train.reshape(len(x_train), img_height, img_width, 1)
    x_test = x_test.reshape(len(x_test), img_height, img_width, 1)
    x_val = x_val.reshape(len(x_val), img_height, img_width, 1)

    # Define and compile CNN model
    model = define_model()
    model.compile(loss="mean_squared_error", optimizer='adam', \
            metrics=["accuracy"])
    model.load_weights('./training/train_v302/train_v302_model.h5')
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', \
            mode = 'max', patience = es_patience)

    # Learning network
    hist = model.fit(x_train, y_train, batch_size = batch_size, \
            epochs = epochs, validation_data = (x_val, y_val),  \
            callbacks = [es])
    model.summary()

    # Test data predictions
    test_loss = model.evaluate(x_test, y_test)[0]
    test_accuracy = model.evaluate(x_test, y_test)[1]
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_accuracy)
    predictions = model.predict(x_test)

    # Saving model
    print('Do you want to save this model? y/n')
    answer = input()
    if answer == 'y':
        print('Input network name:')
        filename = input()
        if not os.path.exists('./training'):
            os.mkdir('./training')
        save_nn_model(filename, model, hist)
        model_info_to_txt(filename, model, test_loss, test_accuracy)
        plot_predictions(filename, x_test, y_test, predictions, \
            output_labels)
        plot_model_loss(filename, hist)
        plot_model_accuracy(filename, hist)


if __name__ == '__main__':
    main()
