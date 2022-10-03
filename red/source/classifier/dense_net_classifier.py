from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf


image_gen = ImageDataGenerator(rescale=1. / 255.)

train_flow = image_gen.flow_from_directory(
    'sampled_data/train/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

image_gen1 = ImageDataGenerator(rescale=1. / 255.)

valid_flow = image_gen1.flow_from_directory(
    'sampled_data/valid/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

dense_net = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
dense_net.trainable = False
model = tf.keras.Sequential()

model.add(dense_net)
model.add(keras.layers.GlobalAveragePooling2D())

for _ in range(2):
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics='accuracy')
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

model.fit(
    train_flow,
    batch_size=256,
    validation_data=valid_flow,
    epochs=5,
    callbacks=[early_stopping_callback]
)
model.save('dense_net_trained.h5')
