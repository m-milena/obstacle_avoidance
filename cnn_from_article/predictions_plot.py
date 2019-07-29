import matplotlib.pyplot as plt
from skimage import color
import numpy as np

class_names = ['Full right', 'Half right', 'Straightforward', 'Half letf', 'Full left']

def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img[:,:,0], cmap="gray", interpolation='nearest', vmin=0, vmax=1)
	predicted_label = np.argmax(predictions_array)
	if predicted_label == np.argmax(true_label):
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[np.argmax(true_label)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(5), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)
  
	thisplot[predicted_label].set_color('red')
	thisplot[np.argmax(true_label)].set_color('blue')

def plot_example_predictions(filename, x_test, y_test, predictions):
	num_rows = 5
	num_cols = 4
	num_images = num_rows*num_cols
	plt.figure(figsize=(2*2*num_cols, 2*num_rows))
	for i in range(num_images):
		plt.subplot(num_rows, 2*num_cols, 2*i+1)
		plot_image(i, predictions, y_test, x_test)
		plt.subplot(num_rows, 2*num_cols, 2*i+2)
		plot_value_array(i, predictions, y_test)
	plt.savefig('training/' + filename + '/' + filename+'_test_prediction.png',dpi=400)
	plt.show()
