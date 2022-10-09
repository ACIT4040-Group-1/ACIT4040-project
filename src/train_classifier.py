from keras_preprocessing.image import ImageDataGenerator

from common.DataLoader import DataLoader
from src.red.classifier.model import get_model
from tensorflow import keras

if __name__ == '__main__':
    image_gen = ImageDataGenerator(rescale=1. / 255.)
    valid_images = image_gen.flow_from_directory('red/training_data/valid', target_size=(224, 224), batch_size=1,
                                                 class_mode='binary')

    image_gen = ImageDataGenerator(rescale=1. / 255.)
    train_images = image_gen.flow_from_directory('red/training_data/train', target_size=(224, 224), batch_size=1,
                                                 class_mode='binary')
    model = get_model()
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    model.fit(
        train_images,
        batch_size=256,
        validation_data=valid_images,
        epochs=15,
        callbacks=[early_stopping_callback]
    )
    model.save('red/classifier/trained_models/red_classifier.h5')
