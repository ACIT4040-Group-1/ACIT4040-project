import numpy as np
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from scipy.linalg import sqrtm


def scale_images(images, new_shape):
    """
    Scales a list of images to a new size
    :param images: Images to be reshaped
    :param new_shape: The new shape
    :return: Array of images
    """
    images = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images.append(new_image)

    return np.asarray(images)


def calculate_fid(images1, images2, input_shape=(299, 299, 3)):
    """
    Frechet Inception Distance, as implemented by Jason Brownlee, 08.30.19.
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    :return:
    """
    images1 = images1.astype('float32')
    images2 = images1.astype('float32')

    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
    images1 = scale_images(images1, input_shape)
    images2 = scale_images(images2, input_shape)

    act1 = model.predict(images1)
    act2 = model.predict(images2)

    mean1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mean2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    sum_squared_diff = np.sum((mean1, mean2)**2.0)
    cov_mean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = sum_squared_diff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)
    return fid
