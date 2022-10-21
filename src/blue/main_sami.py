
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

#import sys
#sys.path.insert(0,'C:\\Users\\saman\\github\\ACIT4040-project\\src\\common')
#sys.path.insert(0,'C:\\Users\\saman\\github\\ACIT4040-project\\src')
#sys.path.insert(0,'C:\\Users\\saman\\github\\ACIT4040-project\\config')
#print(sys.path)
from src.common.DataLoader import  DataLoader
from src.common.utils import get_config, copy_config

class ConvBlocks:
    @staticmethod
    def BNConv(x, filters, kernel_size, strides, l2_weight=1e-4, has_act=True):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                          padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_weight))(x)
        x = layers.BatchNormalization()(x)
        if has_act:
            x = layers.ReLU()(x)
        return x

    @staticmethod
    def SepConv(x, filters, kernel_size, strides=(1, 1), l2_weight=1e-4, relu='front'):
        assert (relu == 'front') or (relu == 'back') or (relu == None)
        if relu == 'front':
            x = layers.ReLU()(x)
        x = layers.SeparableConv2D(filters, kernel_size=kernel_size, strides=strides,
                                   padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_weight))(x)
        x = layers.BatchNormalization()(x)
        if relu == 'back':
            x = layers.ReLU()(x)
        return x

    @classmethod
    def SepConvMaxPoolBlock(cls, x_in, filters, l2_weight=1e-4, front_relu=True):
        assert type(front_relu) == type(True)
        assert len(filters) == 2
        x = cls.SepConv(x_in, filters[0], (3, 3), l2_weight=l2_weight, relu='front' if front_relu else None)
        x = cls.SepConv(x, filters[1], (3, 3), l2_weight=l2_weight, relu='front')
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x_skip = cls.BNConv(x_in, filters[1], kernel_size=(1, 1), strides=(2, 2), l2_weight=l2_weight, has_act=False)
        x = layers.Add()([x_skip, x])
        return x

    @classmethod
    def SepConvBlock(cls, x_in, filters, repeats, l2_weight=1e-4):
        for n in range(repeats):
            x = cls.SepConv(x_in, filters, kernel_size=(3, 3), strides=(1, 1), l2_weight=1e-4, relu='front')
            x = cls.SepConv(x, filters, kernel_size=(3, 3), strides=(1, 1), l2_weight=1e-4, relu='front')
            x = cls.SepConv(x, filters, kernel_size=(3, 3), strides=(1, 1), l2_weight=1e-4, relu='front')
            x_in = layers.Add()([x_in, x])
        return x_in


def XceptionNet(input_shape=(299, 299, 3), n_classes=1000,
                entry_flow=(('conv', 32, True), ('conv', 64, False), ('sconv_pool', 128, 128), ('sconv_pool', 256, 256),
                            ('sconv_pool', 728, 728)),
                middle_flow_repeat=9,
                exit_flow=(('sconv_pool', 728, 1024), ('sconv', 1536), ('sconv', 2048)),
                dropout_rate=0.5, l2_weight=1e-4):
    x_in = layers.Input(shape=input_shape)
    x = x_in
    # Entry flow
    for n, (block_type, attr1, attr2) in enumerate(entry_flow):
        if n == 0 and block_type != 'conv':
            raise Exception('entry flow must begin with a "conv" !')
        if block_type == 'conv':
            x = ConvBlocks.BNConv(x, attr1, (3, 3), 2 if attr2 else 1, l2_weight, True)
        elif block_type == 'sconv_pool':
            x = ConvBlocks.SepConvMaxPoolBlock(x, (attr1, attr2), l2_weight,
                                               front_relu=(entry_flow[n - 1][0] != 'conv'))
        else:
            raise Exception('entry flow should be built via "conv" and "sconv_pool" only')

    # middle flow
    x = ConvBlocks.SepConvBlock(x, entry_flow[-1][-1], repeats=middle_flow_repeat, l2_weight=1e-4)

    # exit flow
    for block in exit_flow:
        if block[0] == 'sconv_pool':
            x = ConvBlocks.SepConvMaxPoolBlock(x, (block[1], block[2]), l2_weight, front_relu=True)
        elif block[0] == 'sconv':
            x = ConvBlocks.SepConv(x, block[1], 3, 1, l2_weight, 'back')
        else:
            raise Exception('exit flow should be built via "sconv" and "sconv_pool" only')
    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)
    x_out = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=x_in, outputs=x_out)
    return model


if __name__ == '__main__':
    # model = XceptionNet()
    # model.summary()
    # plot_model(model, 'model.png', show_shapes = True)

    model = XceptionNet(input_shape=(256, 256, 3), n_classes=2,
                        entry_flow=(('conv', 32, False), ('conv', 64, False), ('sconv_pool', 64, 64)),
                        middle_flow_repeat=9,
                        exit_flow=(('sconv_pool', 128, 128), ('sconv', 256), ('sconv', 256)),
                        dropout_rate=0.5, l2_weight=1e-4)
    #model.summary()
    #plot_model(model, 'model.png', show_shapes=True)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    DL = DataLoader()
    test_ds = DL.get_data("test")

    train_ds = DL.get_data("train")

    print("Fit model on training data")
    history = model.fit(
        train_ds,
        batch_size=64,
        epochs=10000,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(test_ds),
    )
    history.history

    print("Evaluate on test data")
    results = model.evaluate(test_ds, batch_size=128)
    print("test loss, test acc:", results)
