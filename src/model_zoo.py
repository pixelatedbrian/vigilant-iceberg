import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, ZeroPadding2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import cv2
import keras


def gmodel(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 3)):
    # Build keras model

    image_model=Sequential()

    # CNN 1
    image_model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=input_shape))
    image_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    image_model.add(Dropout(drop_out))

    # CNN 2
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(drop_out))

    # CNN 3
    image_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(drop_out))

    # CNN 4
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    image_model.add(Dropout(drop_out))

    # You must flatten the data for the dense layers
    image_model.add(Flatten())

    # Image input encoding
    image_input = Input(shape=(75,75,3))
    encoded_image = image_model(image_input)

    # Inc angle input
    inc_angle_input = Input(shape=(1,))

    # Combine image and inc angle
    combined= keras.layers.concatenate([encoded_image,inc_angle_input])


    dense_model = Sequential()

    # Dense 1
    dense_model.add(Dense(512, activation='relu',input_shape=(257,)))
    dense_model.add(Dropout(drop_out))

    # Dense 2
    dense_model.add(Dense(256, activation='relu'))
    dense_model.add(Dropout(drop_out))

    # Output
    dense_model.add(Dense(1, activation="sigmoid"))

    output = dense_model(combined)

    # Final model
    combined_model= Model(inputs=[image_input,inc_angle_input],outputs= output)

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return combined_model


def gmodel2(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 3)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

    # X_inc_angle = Input(shape=(1,))
    X_inc_angle = Input(shape=(1,))

    ##############
    ### CONV 0 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv0")(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), strides=(2, 2), name="max_pool0")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 1 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv1")(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool1")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 2 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(256, (3, 3), strides=(1, 1), name="conv2")(X)
    X = BatchNormalization(axis=3, name='bn_conv2')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool2")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 3 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(512, (3, 3), strides=(1, 1), name="conv3")(X)
    X = BatchNormalization(axis=3, name='bn_conv3')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool3")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    # FLATTEN -> FC
    X = Flatten()(X)

    ########################
    ### Add in inc_angle ###
    ########################
    X = concatenate([X, X_inc_angle])

    # FC
    X = Dense(512, name="fc0")(X)
    X = BatchNormalization(name="bn_fc0")(X)
    X = Activation("relu")(X)
    X = Dropout(drop_out)(X)

    # FC
    X = Dense(256, name="fc1")(X)
    X = BatchNormalization(name="bn_fc1")(X)
    X = Activation("relu")(X)
    X = Dropout(drop_out)(X)

    X = Dense(1, activation="sigmoid", name='fc2')(X)

    model = Model(inputs=[X_input, X_inc_angle], outputs=X, name="gmodel2")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
