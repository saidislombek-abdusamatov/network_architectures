from tensorflow.keras.layers import *
from tensorflow.keras import Model

def identity_block(x, filters):
   
    f1, f2, f3 = filters

    x1 = x
   
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, x1])

    return x

def convolutional_block(x, filters, s=2):

    f1, f2, f3 = filters

    x1 = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s))(x1)
    x1 = BatchNormalization()(x1)

    x = Add()([x, x1])
    X = Activation('relu')(x)

    return X

def ResNet50(input_shape=(224, 224, 3), num_classes=1000):

    x_input = Input(input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = convolutional_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = convolutional_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = convolutional_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = convolutional_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = GlobalAveragePooling2D()(x)
    x_output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=x_input, outputs=x_output, name='ResNet50')

    return model