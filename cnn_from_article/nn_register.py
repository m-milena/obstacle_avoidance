import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import color

def __plot_image(i, predictions_array, true_label, img, class_names):
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


def __plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(true_label)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
  
    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')


def plot_predictions(filename, x_test, y_test, predictions, class_names):
    num_rows = 5
    num_cols = 4
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        __plot_image(i, predictions, y_test, x_test, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        __plot_value_array(i, predictions, y_test)
    plt.savefig('training/' + filename + '/' + filename+'_test_prediction.png',dpi=400)
    plt.show()


def save_nn_model(filename, model, hist):
    # Creating NN folder
    os.mkdir('./training/' + filename)
    model_path = './training/' + filename + '/' + filename
    # Saving model to *.model and *.h5 files
    model.save(model_path + '_model.model')
    model.save(model_path + '_model.h5')
    # Saving model to *.json file
    model_json = model.to_json()
    with open(model_path + '_model.json', 'w') as json_file:
        json_file.write(model_json)
    # Saving learning history
    hist_df = pd.DataFrame(hist.history) 
    with open(model_path + '_logg.csv', 'w') as f:
        hist_df.to_csv(f)


def plot_model_loss(filename, hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig('training/' + filename + '/' + filename + \
            '_loss_graph.png',dpi=400)
    plt.show()


def plot_model_accuracy(filename, hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig('training/' + filename + '/' + filename + \
            '_acc_graph.png',dpi=400)
    plt.show()


def model_info_to_txt(filename, model, test_loss, test_accuracy):
    file_path = 'training/' + filename + '/' + filename + \
            '_model_description.txt'
    file = open(file_path, 'w')
    text = 'Model description of ' + filename +'\n' + 65*'_' + \
            '\n\nTest loss: ' + str(test_loss) + '\nTest accuracy: '\
             + str(test_accuracy) + '\n'
    file.write(text)
    with open(file_path, 'a') as file: 
        model.summary(print_fn = lambda x: file.write(x + '\n'))
    file.close()

