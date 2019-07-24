import matplotlib.pyplot as plt

def plot_model_loss(hist, answer, filename):
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper right')
	if answer == 'y':
		plt.savefig('training/' + filename + '/' + filename + '_loss_graph.png',dpi=400)
		plt.show()

def plot_model_accuracy(hist, answer, filename):
	plt.plot(hist.history['acc'])
	plt.plot(hist.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='lower right')
	if answer == 'y':
		plt.savefig('training/' + filename + '/' + filename + '_acc_graph.png',dpi=400)
		plt.show()
