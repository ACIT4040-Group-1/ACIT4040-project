from tensorflow import keras
model = keras.models.load_model('src/blue/trained_models/akselnet/2022-10-10 17-49-23/best_model')
print(model.summary())