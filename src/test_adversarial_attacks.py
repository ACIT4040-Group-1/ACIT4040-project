from src.red.AdversarialFramework import AdversarialFramework
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    model = keras.models.load_model('red/classifier/trained_models/dense_net_trained.h5')

    AF = AdversarialFramework(model=model)
    images = AF.load_images()

    AF.evaluate_model(images=images)

    # Create new FGSM images of input images.
    images = AF.apply_attack(method='FGSM', images=images)

    # Loading already saved attack images (if they exist)
    # images = AF.apply_attack(method='FGSM', create_new_images=False)

    AF.evaluate_model(images=images)
