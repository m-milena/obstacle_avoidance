# load dataset
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd

dataset_input = []
dataset_output = []
images = []
folder = "./dataset"
img_width = 160
img_height = 120

images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
random.shuffle(images)

for i in images:
# output prepare
	direction = int(i[1:i.find('_')])
	output = direction*[0] + [1] + (5-direction-1)*[0]
	dataset_output.append(output)
# input prepare
	img = load_img(folder +'/' + i, color_mode="grayscale")
	img.thumbnail((img_width, img_height))
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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense

# CNN model
model = Sequential([
	Conv2D(32, (5,5), padding = 'same', input_shape = (img_height, img_width, 1), activation = 'relu'),
	MaxPooling2D(pool_size = (2,2), strides = 2),
	Conv2D(32, (5,5), padding = 'same', activation = 'relu'),
	AveragePooling2D(pool_size = (2,2), strides = 2),
	Conv2D(64, (5,5), padding = 'same', activation = 'relu'),
	AveragePooling2D(pool_size = (2,2), strides = 2),
	Flatten(),
	Dense(256, activation='linear'),
	Dense(128, activation='linear'),
	Dense(5, activation='linear')
])

# training model
model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])

hist = model.fit(x_train, y_train, batch_size=30, epochs=10, validation_data = (x_val, y_val))
model.summary()

loss = model.evaluate(x_test, y_test)[0]
accuracy = model.evaluate(x_test, y_test)[1]
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

# saving model
print('Do you want to save this model? y/n')
answer = input()
if answer == 'y':
	print('Input filename:')
	filename = input()
	model.save(filename+'.model')
	model.save(filename+'.h5')
	model_json = model.to_json()
	with open(filename+'.json', 'w') as json_file:
		json_file.write(model_json)
	hist_df = pd.DataFrame(hist.history) 
	with open(filename+'_logg.csv', 'w') as f:
    		hist_df.to_csv(f)

# visualisation
import train_visualisation
train_visualisation.plot_model_loss(hist, answer, filename)
train_visualisation.plot_model_accuracy(hist, answer, filename)
