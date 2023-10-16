
from tensorflow.keras.layers import *
from tensorflow.keras import Model

def dense_block(x, iteration):
    for _ in range(iteration):
        y = x
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (1,1), strides=(1,1), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3,3), strides=(1,1), padding="same", use_bias=False)(x)
        x = concatenate([x, y])
    return x

def transition_layer(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(x.shape[-1]// 2, (1,1), strides=(1,1), padding="same", use_bias=False)(x)
    x = AveragePooling2D((2,2), strides=(2,2), padding="same")(x)
    return x

def DenseNet121(input_shape=(224,224,3), num_classes=1000):
    x_input = Input(input_shape)
    x = Conv2D(64, (7,7), strides=(2,2), padding="same", use_bias=False)(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3,3), strides=(2,2), padding="same")(x)
    
    for iteration in [6, 12, 24, 16]:
        y = dense_block(x, iteration)
        x = transition_layer(y)
    
    x = GlobalAveragePooling2D()(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=x_input, outputs=x_output, name='DenseNet121')
    return model