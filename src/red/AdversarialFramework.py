import json
import os

import numpy as np
from PIL import Image
from keras.losses import MSE
from sklearn import metrics
import tensorflow as tf
from tqdm import tqdm


class AdversarialFramework:
    def __init__(self, model):
        self.model = model

    def apply_attack(self, method, images):
        methods = {
            'FSGM': self.fast_gradient_signed_method,
        }

        attack_method = methods.get(method)

        if not attack_method:
            raise Exception(f'Method {method} is not supported yet')

        number_of_images = len(images)
        adversarial_images = []

        print(f'Applying method {method}..')
        for step, (image, label) in tqdm(enumerate(images)):
            if step >= number_of_images:
                break

            adversarial_image = attack_method(image, label)
            adversarial_images.append(adversarial_image)

        if not os.path.exists(f'adversarial_images/{method}'):
            os.mkdir(f'adversarial_images/{method}')

        return adversarial_images

    def evaluate_model(self, images):
        print('Evaluation model..')
        predictions = self.model.predict(images)
        real_labels = images.classes

        real_images, fake_images, classified, misclassified = 0, 0, 0, 0

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
                misclassified += 1
            elif predicted_fake and real_image:
                misclassified += 1
            else:
                classified += 1

        roc_auc_score = metrics.roc_auc_score(real_labels, predictions)
        ap_score = metrics.average_precision_score(real_labels, predictions)

        print('RESULTS: ')
        print('Number of images: ', len(predictions))
        print(f'ROC AUC Score: {roc_auc_score}')
        print(f'AP Score: {ap_score}')
        print(f'Real images: {real_images}')
        print(f'Fake images: {fake_images}')
        print(f'Classified correctly: {classified}')
        print(f'Misclassified: {misclassified}')
        # print(metrics.classification_report(y_test, y_pred > 0.5))

    def fast_gradient_signed_method(self, image, label, eps=2 / 255.0):
        # cast the image
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
            return adversary
