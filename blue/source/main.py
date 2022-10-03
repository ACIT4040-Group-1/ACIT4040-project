
from time import time
from datetime import datetime
from numpy import save
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
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
    print("Loading data..")
    DL = DataLoader()
    train_ds = DL.get_data("train")
    test_ds = DL.get_data("test")
    print("Loading comp.")
    model = get_model()

    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5)
    # mm-dd-hh-mm-config['model_name']
    tensorboard_callback = TensorBoard(log_dir=os.path.join(config['tensorboard']['log_dir'],'akselnet', datetime.now().strftime('%Y-%m-%d %H-%M-%S')))
    save_model_callback = ModelCheckpoint(filepath=os.path.join(config['model_dir']), monitor="val_accuracy", save_best_only= True)

    model.fit(x=train_ds,
              batch_size=config['batch_size'],
              validation_data=test_ds,
              epochs=1,
              callbacks=[early_stopping_callback, tensorboard_callback, save_model_callback])
