import os
import re
from time import time
import random
import math
import datetime

import cv2
import numpy as np

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers import Dense, Dropout, concatenate, SpatialDropout2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import optimizers
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from nn_register import save_nn_model, plot_model_loss, plot_model_accuracy
from nn_register import model_info_to_txt

data_file = 'dataset_generated.txt'
batch_size = 100
epochs = 50


def read_data():
    with open(data_file) as myfile:
        line = myfile.readline()
        input_data = []
        output_data = []
        while line:
            line = line.replace("\n","")
            fields = line.split(',')
            input_data_str = fields[1:4]
            output_data_str = fields[4:]
            input_data_row = [float(i) for i in input_data_str]
            output_data_row = [float(i) for i in output_data_str]
# dx, dy, dyaw
            if output_data_row[1] > 0:
                output = [0, 0, 1, 0]
            elif output_data_row[1] < 0:
                output = [1, 0, 0, 0]
            elif not output_data_row[0] and not output_data_row[1]:
                output = [0, 0, 0, 1]
            elif output_data_row[0] != 0:
                output = [0, 1, 0, 0]

            line = myfile.readline()
            input_data.append(input_data_row)
            output_data.append(output)
    return np.array(input_data), np.array(output_data)

def define_model():
    model = Sequential([
        Dense(64, activation = 'relu', input_shape=(3,)),
        Dropout(0.3),
        Dense(64, activation = 'relu'),
        Dense(4, activation = 'sigmoid')
    ])
    return model

def main():
    input_data, output_data = read_data()
    x_train, x_val_test, y_train, y_val_test = train_test_split(input_data, output_data, test_size = 0.3)
    x_val, x_test, y_val, y_test = train_test_split(\
                x_val_test, y_val_test, test_size = 0.5)
    model = define_model()
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=adam, \
            metrics=["accuracy"])

    tensorboard = TensorBoard(log_dir="logs\{}".format(time()), histogram_freq=0)
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint, tensorboard]
    hist = model.fit(x_train, y_train, batch_size = batch_size, \
            epochs = epochs, validation_data = (x_val, y_val),  \
            callbacks = callbacks)
    model.summary()

    # Test data predictions
    model.load_weights("weights.best.hdf5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_loss = model.evaluate(x_test, y_test)[0]
    test_accuracy = model.evaluate(x_test, y_test)[1]
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_accuracy)
    predictions = model.predict(x_test)
    print('Input network name:')
    filename = input()
    if not os.path.exists('./training'):
        os.mkdir('./training')
    save_nn_model(filename, model, hist)
    model_info_to_txt(filename, model, test_loss, test_accuracy)
    plot_model_loss(filename, hist)
    plot_model_accuracy(filename, hist)



if __name__ == '__main__':
    main()
