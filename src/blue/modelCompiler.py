import tensorflow as tf
from keras import Sequential
import quality_feature_iqm as iqm
import quality_feature_iqa as iqa
from keras.layers import Dense, Dropout, activation
from src.common.utils import get_config

config = get_config()

def get_exampleNet():
    net_config = config['models']['exampleNet']
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape= net_config['input_shape'],
        include_top= False,
        weights= 'imagenet',
        pooling= 'avg')
    model = Sequential()
    model.add(pretrained_model)
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model

def get_akselnet():
    net_config = config['models']['akselnet']
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape=net_config['input_shape'],
        include_top=False,
        weights='imagenet',
        pooling='avg')
    model = Sequential()
    model.add(pretrained_model)
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()
    print(net_config['loss'])
    print(net_config['metrics'])
    model.compile(optimizer=optimizer,
                  loss=net_config['loss'],
                  metrics=net_config['metrics'])
    return model

def get_syedanet():
    net_config = config['models']['syedanet']
    batch = dl.get_data()
    features_set = []
    for i in batch:
        feature = iqa.compute_msu_iqa_features(rgbImage=i) + iqm.compute_quality_features(
            image=i)
        features_set.append(feature)
    # input (256,256,3)
    # data loader 32,256,256,3
    model = Sequential()
    model.add(features_set)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1), kernel_regularizer='l2',activation='linear')
    optimizer = tf.keras.optimizers.Adam()
    print(net_config['loss'])
    print(net_config['metrics'])
    model.compile(optimizer = optimizer, loss = net_config['loss'], metrics = net_config['metrics'])
    return model

'''
def get_syedasvcnet():
    net_config = config['model']['syedasvcnet']
    features = iqa.compute_msu_iqa_features(rgbImage= 'input_shape') + iqm.compute_quality_features(image= 'input_shape')
    #svc = svm.SVC(C = [2**P for P in range(-3, 14, 2)], kernel =  'rbf')
    #clf = svc.fit(dataX, DataY) # svm
'''