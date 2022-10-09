from common.DataLoader import DataLoader
from src.red.classifier.model import get_model
from tensorflow import keras
if __name__ == '__main__':

    DL = DataLoader()
    train_dataset = DL.get_data("train")
    test_dataset = DL.get_data("test")
    model = get_model()
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    model.fit(
        train_dataset,
        batch_size=256,
        validation_data=test_dataset,
        epochs=5,
        callbacks=[early_stopping_callback]
    )
    model.save('dense_net_trained.h5')
