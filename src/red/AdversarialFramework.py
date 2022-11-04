import os
import shutil

import keras_preprocessing
import numpy as np
from keras.losses import MSE
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')


class AdversarialFramework:
    def __init__(self, model):
        self.model = model

    def apply_attack(self, method, images=None, create_new_images=True):
        root_path = "red/adversarial/images/" + method
        methods = {
            'FGSM': self.fast_gradient_signed_method,
        }
        attack_method = methods.get(method)

        if not attack_method:
            raise Exception(f'Method {method} is not supported yet')

        if not create_new_images:
            if not os.path.exists(root_path):
                raise Exception(f'Could not find images for method {method}')

            return ImageDataGenerator(rescale=1. / 255.).flow_from_directory(root_path,
                                                                             target_size=(224, 224),
                                                                             batch_size=1,
                                                                             class_mode='binary', seed=1)

        if not images:
            raise Exception('Missing input images to apply attack')

        # Create directory for new images
        if not os.path.exists(root_path):
            os.mkdir(root_path)
            os.mkdir(f'{root_path}/real')
            os.mkdir(f'{root_path}/fake')
        else:
            shutil.rmtree(root_path)

        number_of_images = len(images)

        print(f'Applying method {method}..')
        for step, (image, label) in tqdm(enumerate(images)):
            if step >= number_of_images:
                break
            label = label.numpy()
            adversarial_image = attack_method(image, label)

            real_image = True if label == [1.] else False

            adversarial_image = normalize(adversarial_image)

            plt.title(f'True label: {label}')
            # Save image
            if real_image:
                plt.imsave(f'{root_path}/real/{step}.jpg', adversarial_image)
            else:
                plt.imsave(f'{root_path}/fake/{step}.jpg', adversarial_image)

        return ImageDataGenerator(rescale=1. / 255.).flow_from_directory(root_path,
                                                                         target_size=(224, 224),
                                                                         batch_size=1,
                                                                         class_mode='binary', seed=1)

    def evaluate_model(self, images):
        print('Evaluating model..')
        predictions = self.model.predict(images)

        if isinstance(images, keras_preprocessing.image.directory_iterator.DirectoryIterator):
            real_labels = images.classes
        else: #BatchDataset from DataLoader
            real_labels = [x[1][0] for x in list(images.as_numpy_iterator())]

        real_images, fake_images, classified, misclassified_real, misclassified_fake, misclassified = 0, 0, 0, 0, 0, 0

        for prediction, real_label in zip(predictions, real_labels):
            predicted_real = prediction[0] > 0.5
            predicted_fake = prediction[0] < 0.5
            fake_image = True if real_label == [0.] else False
            real_image = True if real_label == [1.] else False

            if real_image:
                real_images += 1

            if fake_image:
                fake_images += 1

            if predicted_real and fake_image:
                misclassified_real += 1
                misclassified += 1
            elif predicted_fake and real_image:
                misclassified_fake += 1
                misclassified += 1
            else:
                classified += 1

        roc_auc_score = metrics.roc_auc_score(real_labels, predictions)
        ap_score = metrics.average_precision_score(real_labels, predictions)

        print('\n RESULTS: ')
        print('Number of images: ', len(predictions))
        print(f'ROC AUC Score: {roc_auc_score}')
        print(f'AP Score: {ap_score}')
        print(f'Real images: {real_images}')
        print(f'Fake images: {fake_images}')
        print(f'Classified correctly: {classified}')
        print(f'Misclassified: {misclassified}')
        print(f'Real image classified as false: {misclassified_fake}')
        print(f'Fake image classified as real: {misclassified_real}')
        # print(metrics.classification_report(y_test, y_pred > 0.5))

    def fast_gradient_signed_method(self, image, label, eps=0.1):
        image = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:
            # explicitly indicate that our image should be tacked for
            # gradient updates
            tape.watch(image)
            # use our model to make predictions on the input image and
            # then compute the loss
            pred = self.model(image)
            loss = MSE(label, pred)

            # calculate the gradients of loss with respect to the image, then
            # compute the sign of the gradient
            gradient = tape.gradient(loss, image)
            signed_gradient = tf.sign(gradient)
            # construct the image adversary
            adversary = (image + (signed_gradient * eps)).numpy()
            # return the image adversary to the calling function

            return adversary[0]

    def load_images(self, path='data/test_data'):
        image_gen = ImageDataGenerator(rescale=1. / 255.)
        images = image_gen.flow_from_directory(path,
                                               target_size=(224, 224),
                                               batch_size=1,
                                               class_mode='binary',
                                               seed=1)
        return images


def normalize(x):
    return np.array(
        (x - np.min(x)) / (np.max(x) - np.min(x)))
