# load dataset
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
	img = load_img(folder +'/' + i)
	img.thumbnail((img_width, img_height))
	img = img_to_array(img)a
	img = img / 255.0;
	dataset_input.append(img)	

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

# TODO: learn cnn
model.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])
#hist = model.fit(x_train, y_train, batch_size=30, epochs=80, validation_data = (x_val, y_val))

