import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, ZeroPadding2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, MaxPooling2D

from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import cv2
import keras

'''
    v2

    Modified to not take in/concatenate inc_angle (making things worse?)
    Reduce color channels to 2 because the 3rd channel right now usually just
        involes a really lame transformtion of some kind.
'''


def model1(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 2)):
    # Build keras model

    image_model = Sequential()

    # CNN 1
    image_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
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

    # Dense 1
    image_model.add(Dense(512, activation='relu',input_shape=(257,)))
    image_model.add(Dropout(drop_out))

    # Dense 2
    image_model.add(Dense(256, activation='relu'))
    image_model.add(Dropout(drop_out))

    # Output
    image_model.add(Dense(1, activation="sigmoid"))

    # Image input encoding
    image_input = Input(shape=(75, 75, 2))
    encoded_image = image_model(image_input)

    output = image_model(encoded_image)

    # Final model
    combined_model = Model(inputs=image_input, outputs=output)

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return combined_model


def model2(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 2)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

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

    model = Model(inputs=X_input, outputs=X, name="model2")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def model3(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 2)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

    ##############
    ### CONV 0 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv0")(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = LeakyReLU(alpha=0.3)(X)

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
    X = LeakyReLU(alpha=0.3)(X)

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
    X = LeakyReLU(alpha=0.3)(X)

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
    X = LeakyReLU(alpha=0.3)(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool3")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    # FLATTEN -> FC
    X = Flatten()(X)

    # FC
    X = Dense(512, name="fc0")(X)
    X = BatchNormalization(name="bn_fc0")(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(drop_out)(X)

    # FC
    X = Dense(256, name="fc1")(X)
    X = BatchNormalization(name="bn_fc1")(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(drop_out)(X)

    X = Dense(1, activation="sigmoid", name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name="model3")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def model4(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 2)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

    ################
    ### CONV 1_1 ###
    ################

    # CONV -> BN -> RELU
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv1_1")(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1_1')(X)
    X = LeakyReLU(alpha=0.3)(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ################
    ### CONV 1_2 ###
    ################

    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv1_2")(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1_2')(X)
    X = LeakyReLU(alpha=0.3)(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), strides=(2, 2), name="max_pool1")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ################
    ### CONV 2_1 ###
    ################

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv2_1")(X)
    X = BatchNormalization(axis=3, name='bn_conv2_1')(X)
    X = LeakyReLU(alpha=0.3)(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ################
    ### CONV 2_2 ###
    ################

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv2_2")(X)
    X = BatchNormalization(axis=3, name='bn_conv2_2')(X)
    X = LeakyReLU(alpha=0.3)(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool2")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 3 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(256, (3, 3), strides=(1, 1), name="conv3")(X)
    X = BatchNormalization(axis=3, name='bn_conv3')(X)
    X = LeakyReLU(alpha=0.3)(X)

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
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(drop_out)(X)

    # FC
    X = Dense(256, name="fc1")(X)
    X = BatchNormalization(name="bn_fc1")(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(drop_out)(X)

    X = Dense(1, activation="sigmoid", name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name="model4")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def tiny_model(drop_out=0.20, learning_rate=0.01, lr_decay=0, input_shape=(75, 75, 2)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

    # zero padding: pad border of X_input with zeros
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU
    X = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name="max_pool")(X)
    X = Dropout(drop_out)(X)

    # FLATTEN -> FC
    X = Flatten()(X)

    # FC
    X = Dense(256, name="fc1")(X)
    X = BatchNormalization(name="bn_fc1")(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(drop_out)(X)

    X = Dense(1, activation="sigmoid", name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name="tiny_model")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def model5(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 2)):
    # define input placeholder as a tensor with the shape input_shape.
    # this shape is the shape of the input "image"
    X_input = Input(input_shape)

    # X_inc_angle = Input(shape=(1,))
    X_inc_angle = Input(shape=(1,))

    ##############
    ### CONV 1 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv1_1")(X_input)
    X = Activation("relu")(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv1_2")(X_input)
    X = Activation("relu")(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), name="conv1_3")(X_input)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), strides=(2, 2), name="max_pool1")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 2 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv2_1")(X)
    X = Activation("relu")(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv2_2")(X)
    X = Activation("relu")(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv2_3")(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool2")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 3 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv3")(X)
    # X = BatchNormalization(axis=3, name='bn_conv2')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool3")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 4 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(256, (3, 3), strides=(1, 1), name="conv4")(X)
    # X = BatchNormalization(axis=3, name='bn_conv3')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool4")(X)

    # drop_out
    X = Dropout(drop_out)(X)

    # FLATTEN -> FC
    X = Flatten()(X)

    # FC
    X = Dense(1024, name="fc0")(X)
    # X = BatchNormalization(name="bn_fc0")(X)
    X = Activation("relu")(X)
    X = Dropout(drop_out)(X)

    # FC
    X = Dense(512, name="fc1")(X)
    # X = BatchNormalization(name="bn_fc1")(X)
    X = Activation("relu")(X)
    X = Dropout(drop_out)(X)

    X = Dense(1, activation="sigmoid", name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name="model5")

    optimizer = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
