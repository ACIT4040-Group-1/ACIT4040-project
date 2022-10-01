import tensorflow as tf
from keras.callbacks import EarlyStopping
import os

from blue.source.akselnet import get_akselnet
from common.data_loader import DataLoader
from common.utils import get_config

config = get_config()


def get_model():
    match config['model_name']:
        case 'akselnet':
            return get_akselnet()


if __name__ == "__main__":
    DL = DataLoader()
    train_ds = DL.get_data("train")
    test_ds = DL.get_data("test")

    model = get_model()

    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
    # mm-dd-hh-mm-config['model_name']
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config['tensorboard']['log_dir'], 'akselnet'))

    model.fit(x=train_ds,
              batch_size=config['batch_size'],
              validation_data=test_ds,
              epochs=5,
              callbacks=[early_stopping_callback])
