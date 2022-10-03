from src.red.AdversarialFramework import AdversarialFramework
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    model = keras.models.load_model('red/classifier/trained_models/dense_net_trained.h5')

    image_gen = ImageDataGenerator(rescale=1. / 255.)
    test_images = image_gen.flow_from_directory('red/test_data', target_size=(224, 224), batch_size=1,
                                                class_mode='binary', seed=1, shuffle=False)

    AF = AdversarialFramework(model=model, images=test_images)
    AF.evaluate_model()
    AF.apply_attack('FSGM')
    AF.evaluate_model()
