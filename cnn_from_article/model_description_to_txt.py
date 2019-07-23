def model_description_to_txt(filename, model, batch_size, epochs, test_loss, test_accuracy):
	file = open(filename + '/' + filename + '_model_description.txt','w')
	text = "Model description of" + filename +"\n_________________________________________________________________\nbatch size: " + str(batch_size) + "\nepochs: " + str(epochs) + "\n_________________________________________________________________\n\n" +"Test loss: " + str(test_loss) + "\nTest accuracy: " + str(test_accuracy) + "\n"
	file.write(text)
	with open(filename + '/' + filename + '_model_description.txt','a') as file:
    		model.summary(print_fn=lambda x: file.write(x + '\n'))
	file.close()
