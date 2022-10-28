from src.red.AdversarialFramework import AdversarialFramework
from tensorflow import keras
from src.common.DataLoader import DataLoader

if __name__ == '__main__':

    blue_model = keras.models.load_model('src/blue/trained_models/akselnet/resNet151_detector.h5')
    
    AF = AdversarialFramework(model=blue_model)
    images = DataLoader().get_data('test')

    AF.evaluate_model(images=images)

    # Create new FGSM images of input images.
    #images = AF.apply_attack(method='FGSM', images=images)


    # Loading already saved attack images (if they exist)
    images = AF.apply_attack(method='FGSM', create_new_images=False)

    AF.evaluate_model(images=images)