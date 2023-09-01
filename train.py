from preprocess import get_datasets
from models.basic_model import BasicModel
from models.vgg_model import VGGModel
from models.merged_model import MergedModel
from config import image_size, categories
import matplotlib.pyplot as plt
import time

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
    #'vgg_model': VGGModel,
    #'merged_model': MergedModel,
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    epochs = 15
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.train_model(train_dataset, validation_dataset, epochs)
        print('* Evaluating {}'.format(name))
        model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
        filename = '{}_{}_epochs_timestamp_{}.keras'.format(name, epochs, int(time.time()))
        model.save_model(filename)
        print('* Model saved as {}'.format(filename))
