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

# load dataset
images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
random.shuffle(images)

for i in images:
# output prepare
	direction = int(i[1:i.find('_')])
	output = direction*[0] + [1] + (5-direction-1)*[0]
	dataset_output.append(output)
# input prepare
	img = load_img(folder +'/' + i, color_mode="grayscale")
	img.thumbnail((IMG_WIDTH, IMG_HEIGHT))
	img = img_to_array(img)
	img = img / 255.0;
	dataset_input.append(img)	

dataset_input = np.array(dataset_input, dtype = "float")
dataset_output = np.array(dataset_output)

# Prepare dataset to learn: train-70%, val-15%, test-15%
# split data to training and validation set
from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(dataset_input, dataset_output, test_size = 0.3)
# split data to test and validation set
from sklearn.model_selection import train_test_split
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size = 0.5)

# tf & keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

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
	Dense(128, activation = 'linear'),
	Dense(5, activation = 'linear')
])

# training model
BATCH_SIZE = 16;
EPOCHS = 1;

model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])
hist = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (x_val, y_val))
model.summary()

test_loss = model.evaluate(x_test, y_test)[0]
test_accuracy = model.evaluate(x_test, y_test)[1]
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)

predictions = model.predict(x_test)
from predictions_plot import plot_image, plot_value_array
import matplotlib.pyplot as plt

num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, y_test)
plt.show()

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+20, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+20, predictions, y_test)
plt.show()

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+40, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+40, predictions, y_test)
plt.show()

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+60, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+60, predictions, y_test)
plt.show()

# saving model
print('Do you want to save this model? y/n')
answer = input()
filename = ''
if answer == 'y':
	print('Input filename:')
	filename = input()
	os.mkdir(filename)
	model_description_to_txt(filename, model, BATCH_SIZE, EPOCHS, test_loss, test_accuracy)
	model.save(filename + '/' + filename + '_model.model')
	model.save(filename + '/' + filename + '_model.h5')
	model_json = model.to_json()
	with open(filename + '/' + filename + '_model.json', 'w') as json_file:
		json_file.write(model_json)
	hist_df = pd.DataFrame(hist.history) 
	with open(filename + '/' + filename + '_logg.csv', 'w') as f:
    		hist_df.to_csv(f)

# visualisation
import train_visualisation
train_visualisation.plot_model_loss(hist, answer, filename)
train_visualisation.plot_model_accuracy(hist, answer, filename)
