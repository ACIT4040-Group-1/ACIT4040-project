from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os
import keras
import sklearn.metrics
import itertools
import io

from keras_preprocessing.image import ImageDataGenerator

from config import config
from src.blue import modelCompiler
from src.common.DataLoader import DataLoader
from src.common.utils import copy_config
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def get_model():
    match config['model_name']:
        case 'example_net':
            return modelCompiler.get_exampleNet()
        case 'akselnet':
            return modelCompiler.get_akselnet()
        case 'maryamnet':
            return modelCompiler.get_maryamnet()
        case 'resNet50Detector':
            return modelCompiler.get_res_net50_detector()
        case 'resNet151_detector':
            return modelCompiler.get_res_net151_detector()


log_dir = os.path.join(config['tensorboard']['log_dir'], config['model_name'],
                       datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
model_dir = os.path.join(log_dir, config['model_name']+'.h5')

##loading data
# print("Loading data..")
DL = DataLoader()
train_ds = DL.get_data("train")
test_ds = DL.get_data("test")
val_ds = DL.get_data("valid")
val_labels = np.concatenate([y for x, y in val_ds], axis=0)
class_names = [0, 1]


# print("Loading comp.")


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(val_ds)
    test_pred = np.round(test_pred_raw)
    # print(f"shape val_pred {test_pred.shape}")
    # print(f"shape val_labels {val_labels.shape}")

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(y_true=val_labels, y_pred=test_pred, labels=class_names)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


# Define the per-epoch callback.


if __name__ == "__main__":

    model = get_model()
    if type(model) is None:
        print("No model returned")

    # callbacks
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
    # mm-dd-hh-mm-config['model_name']
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    save_model_callback = ModelCheckpoint(filepath=os.path.join(model_dir), monitor="val_accuracy", save_best_only=True)

    # image_gen = ImageDataGenerator(rescale=1. / 255.)
    # valid_images = image_gen.flow_from_directory('training_data/valid', target_size=(224, 224), batch_size=1,
    #                                              class_mode='binary')
    #
    # image_gen = ImageDataGenerator(rescale=1. / 255.)
    # train_images = image_gen.flow_from_directory('training_data/train', target_size=(224, 224), batch_size=1,
    #                                              class_mode='binary')

    copy_config(log_dir)
    model.fit(x=train_ds,
              batch_size=256,
              validation_data=test_ds,
              epochs=1,
              callbacks=[early_stopping_callback, tensorboard_callback, save_model_callback, cm_callback])
