import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
from model_description_to_txt import model_description_to_txt

dataset_input = []
dataset_output = []
images = []
folder = "./dataset"
IMG_WIDTH = 160
IMG_HEIGHT = 120

# Load dataset
images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
random.shuffle(images)

for i in images:
# Output prepare
	direction = int(i[1:i.find('_')])
	output = direction*[0] + [1] + (5-direction-1)*[0]
	dataset_output.append(output)
# Input prepare
	img = load_img(folder +'/' + i, color_mode="grayscale")
	img.thumbnail((IMG_WIDTH, IMG_HEIGHT))
	img = img_to_array(img)
	img = img / 255.0;
	dataset_input.append(img)	

dataset_input = np.array(dataset_input, dtype = "float")
dataset_output = np.array(dataset_output)

# Prepare dataset to learn: train-70%, val-15%, test-15%
from sklearn.model_selection import train_test_split
# Split data to training and validation set
x_train, x_val_test, y_train, y_val_test = train_test_split(dataset_input, dataset_output, test_size = 0.3)
# Split data to test and validation set
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5)

# tf & keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2
from predictions_plot import plot_example_predictions

# CNN model
model = Sequential([
	Conv2D(32, (5,5), padding = 'same', input_shape = (IMG_HEIGHT, IMG_WIDTH, 1), activation = 'relu'),
	MaxPooling2D(pool_size = (2,2), strides = 2),
	Conv2D(32, (5,5), padding = 'same', activation = 'relu'),
	AveragePooling2D(pool_size = (2,2), strides = 2),
	Conv2D(64, (5,5), padding = 'same', activation = 'relu'),
	AveragePooling2D(pool_size = (2,2), strides = 2),
	Flatten(),
	Dense(256, activation = 'linear'),
	Dropout(0.5),
	Dense(128, activation = 'linear'),
	Dense(5, activation = 'linear')
])

# Training 
BATCH_SIZE = 16;
EPOCHS = 300;

model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])
hist = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (x_val, y_val))
model.summary()

# Testing
test_loss = model.evaluate(x_test, y_test)[0]
test_accuracy = model.evaluate(x_test, y_test)[1]
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)
predictions = model.predict(x_test)

# Save model
print('Do you want to save this model? y/n')
answer = input()
filename = ''
if answer == 'y':
	print('Input filename:')
	filename = input()
	os.mkdir("training/" + filename)
	plot_example_predictions(filename, x_test, y_test, predictions)
	model_description_to_txt(filename, model, BATCH_SIZE, EPOCHS, test_loss, test_accuracy)
	model.save('training/' + filename + '/' + filename + '_model.model')
	model.save('training/' + filename + '/' + filename + '_model.h5')
	model_json = model.to_json()
	with open('training/' + filename + '/' + filename + '_model.json', 'w') as json_file:
		json_file.write(model_json)
	hist_df = pd.DataFrame(hist.history) 
	with open('training/' + filename + '/' + filename + '_logg.csv', 'w') as f:
    		hist_df.to_csv(f)

# Training visualisation
import train_visualisation
train_visualisation.plot_model_loss(hist, answer, filename)
train_visualisation.plot_model_accuracy(hist, answer, filename)
