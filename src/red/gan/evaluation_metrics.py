import numpy as np
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from math import floor


def scale_images(images, new_shape):
    """
    Scales a list of images to a new size
    :param images: Images to be reshaped
    :param new_shape: The new shape
    :return: Array of images
    """
    new_images = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        new_images.append(new_image)

    return np.asarray(new_images)


def calculate_fid(real_images: np.array, fake_images: np.array, input_shape=(299, 299, 3)) -> float:
    """
    Frechet Inception Distance, as implemented by Jason Brownlee, 08.30.19.
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

    Lower score is better.

    :param real_images: Array of real images
    :param fake_images: Array of fake images
    :param input_shape: Input shape of images to the Inception model. Keep default for now
    :return: FID evaluation metric
    """
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)

    real_images = real_images.astype('float32')
    fake_images = real_images.astype('float32')

    real_images = scale_images(real_images, input_shape)
    fake_images = scale_images(fake_images, input_shape)

    real_images = preprocess_input(real_images)
    fake_images = preprocess_input(fake_images)

    act1 = model.predict(real_images)
    act2 = model.predict(fake_images)

    mean1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mean2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    sum_squared_diff = np.sum((mean1 - mean2)**2.0)
    cov_mean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = sum_squared_diff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)
    return fid


def calculate_inception_score(images: np.array, n_splits=10, eps=1E-16, input_shape=(299, 299, 3)) -> (float, float):
    """
    Inception Score, as implemented by Jason Brownlee, 08.28.19.
    https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

    Higher score is better.

    :param images: Array of fake images
    :param n_splits: Number of splits of the data
    :param eps: Epsilon
    :param input_shape: Input shape of images to the Inception model. Keep default for now
    :return: Inception score mean and standard deviation.
    """
    model = InceptionV3()

    images = images.astype('float32')
    images = scale_images(images, input_shape)
    images = preprocess_input(images)

    y_hat = model.predict(images)
    scores = list()
    n_part = floor(images.shape[0] / n_splits)

    for i in range(n_splits):
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = y_hat[ix_start:ix_end]
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)

    is_avg = np.mean(scores)
    is_std = np.std(scores)

    return is_avg, is_std


if __name__ == "__main__":
    pass
