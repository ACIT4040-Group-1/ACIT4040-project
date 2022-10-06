from tensorflow import keras
from src.common.DataLoader import DataLoader
from src.red.classifier.model import get_model

DL = DataLoader()
train_data = DL.get_data('train')
val_data = DL.get_data('valid')

model = get_model()
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

model.fit(
    train_data,
    batch_size=256,
    validation_data=val_data,
    epochs=5,
    callbacks=[early_stopping_callback]
)
model.save('dense_net_trained.h5')
