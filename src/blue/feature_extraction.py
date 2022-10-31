import quality_feature_iqm as iqm
import quality_feature_iqa as iqa
import tensorflow as tf
@tf.function
def calculate_image_features(image):

    '''compute the total features set in images'''

    feature = iqa.compute_msu_iqa_features(image) + iqm.compute_quality_features(image)
    return feature



