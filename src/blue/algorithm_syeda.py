import matplotlib.pyplot as plt
import sklearn.ensemble
from numpy import moveaxis

from src.common.DataLoader import DataLoader
from src.common.utils import get_config, copy_config
import numpy as np
import tensorflow as tf
import quality_feature_iqm as iqm
import quality_feature_iqa as iqa
import pandas as pd



config = get_config()
tf.config.run_functions_eagerly(True)

from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os
import keras
import sklearn.metrics
import itertools
import io
from config import config
from src.blue import modelCompiler
from src.common.DataLoader import DataLoader
from src.common.utils import copy_config
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

log_dir = os.path.join(config['tensorboard']['log_dir'], config['model_name'],
                           datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
model_dir = os.path.join(log_dir, config['model_name'] + '.h5')

## loading data

DL = DataLoader()
train_ds = DL.get_data("train")
test_ds = DL.get_data("test")
val_ds = DL.get_data("valid")
val_labels = np.concatenate([y for x, y in val_ds], axis=0)
class_names = [0, 1]
print("Loading comp.")
train_fs =[]

for i in train_ds:

  image = i[0].numpy()[0, :, :, :]
  #print(image.shape)
  fs1 = iqm.compute_quality_features(image)
  fs2 = iqa.compute_msu_iqa_features(image)
  fs = np.concatenate((fs1,fs2))
  train_fs.append(fs)



train_fs = np.array(train_fs)
print(train_fs.shape)



test_fs = []

for i in test_ds:

  image = i[0].numpy()[0, :, :, :]
  #print(image.shape)
  fs1 = iqm.compute_quality_features(image)
  fs2 = iqa.compute_msu_iqa_features(image)
  fs = np.concatenate((fs1,fs2))
  test_fs.append(fs)


val_fs =[]

for i in val_ds:

  image = i[0].numpy()[0, :, :, :]
  #print(image.shape)
  fs1 = iqm.compute_quality_features(image)
  fs2 = iqa.compute_msu_iqa_features(image)
  fs = np.concatenate((fs1,fs2))
  val_fs.append(fs)


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



import keras.optimizers
import tensorflow as tf

from keras import Sequential, Model
from keras.layers import Dense, Dropout, concatenate, Input

#timesteps = 50
features = Input(shape=(,), name='Features')
model = Sequential()
model.add(Dense(units=256, activation='relu'))(features)
model.add(Dropout(rate=0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='hinge', metrics= 'binary-accuracy')


# callbacks
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
tensorboard_callback = TensorBoard(log_dir=log_dir)
save_model_callback = ModelCheckpoint(filepath=os.path.join(model_dir), monitor="val_accuracy", save_best_only=True)

model.build(input_shape=(256, 256, 3))
print('Model summary: ')
model.summary()

copy_config(log_dir)
model.fit(x=train_fs,
          batch_size=32,
          validation_data=test_fs,
          epochs=15,
          callbacks=[early_stopping_callback, tensorboard_callback, save_model_callback, cm_callback])




#SVM Algorithm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
#DATASET for SVM:

X = train_fs
y = test_fs

# Creating training and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a SVM classifier using SVC class
svm = SVC(kernel= 'rbf', random_state=1, C=0.1)
svm.fit(X_train_std, y_train)

# Mode performance

y_pred = svm.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))



'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm


from  sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier().fit(train_fs, val_fs)

score = clf.score(train_fs,val_fs)
print(score)
'''
# Plot the decision boundary for a non-linear SVM problem
def plot_decision_boundary(model, ax=None):
  if ax is None:
    ax = plt.gca()

  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  # create grid to evaluate model
  x = np.linspace(xlim[0], xlim[1], 30)
  y = np.linspace(ylim[0], ylim[1], 30)
  Y, X = np.meshgrid(y, x)

  # shape data
  xy = np.vstack([X.ravel(), Y.ravel()]).T

  # get the decision boundary based on the model
  P = model.decision_function(xy).reshape(X.shape)

  # plot decision boundary
  ax.contour(X, Y, P,
             levels=[0], alpha=0.5,
             linestyles=['-'])



# plot data and decision boundary
plt.scatter(train_fs[:, 0], train_fs[:, 1], c=val_fs, s=50)
plot_decision_boundary(nonlinear_clf)
plt.scatter(nonlinear_clf.support_vectors_[:, 0], nonlinear_clf.support_vectors_[:, 1], s=50, lw=1, facecolors='none')
'''