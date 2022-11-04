import keras
from src.common.utils import plot_predictions
from src.common.DataLoader import DataLoader

DL = DataLoader()
data = DL.get_data('valid')
model = keras.models.load_model('src/blue/models/resNet151_detector.h5')
# result = model.predict(data)
plot_predictions(data, model, batch=1)
