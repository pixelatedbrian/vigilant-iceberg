#Import the required libraries
import numpy as np
np.random.seed(1338)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt

def build_model(input_shape, nb_classes, X_train, Y_train, X_test, Y_test):
    """"""
    # -- Initializing the values for the convolution neural network

    nb_epoch = 100  # kept very low! Please increase if you have GPU

    batch_size = 32
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # Vanilla SGD
    sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.55))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.55))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=nb_epoch,verbose=1,
              validation_data=(X_test, Y_test))


    #Evaluating the model on the test data
    score, accuracy = model.evaluate(X_train, Y_train, verbose=0)
    print("Train Score: {:2.4f}".format(score))
    print("Train Accuracy: {:1.4f}".format(100.0 * accuracy))
    score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("Test Score: {:2.4f}".format(score))
    print("Test Accuracy: {:1.4f}".format(100.0 * accuracy))

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

def main():
    #Load the training and testing data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_test_orig = X_test

    # correctly process the image data color channels
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 1)

    # preprocess and normalise the data
    X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
    X_test = X_test.reshape((X_test.shape[0],) + shape_ord)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # we know that the grey value is between 0 and 255
    # so this normalizes between those values
    X_train /= 255
    X_test /= 255

    np.random.seed(1338)  # for reproducibilty!!

    # Test data
    X_test = X_test.copy()
    Y = y_test.copy()

    # Converting the output to binary classification(Six=1,Not Six=0)
    Y_test = Y == 6
    Y_test = Y_test.astype(int)

    # Selecting the 5918 examples where the output is 6
    X_six = X_train[y_train == 6].copy()
    Y_six = y_train[y_train == 6].copy()

    # Selecting the examples where the output is not 6
    X_not_six = X_train[y_train != 6].copy()
    Y_not_six = y_train[y_train != 6].copy()

    # Selecting 6000 random examples from the data that
    # only contains the data where the output is not 6
    random_rows = np.random.randint(0,X_six.shape[0],6000)
    X_not_six = X_not_six[random_rows]
    Y_not_six = Y_not_six[random_rows]

    # Appending the data with output as 6 and data with output as <> 6
    X_train = np.append(X_six,X_not_six)

    # Reshaping the appended data to appropraite form
    X_train = X_train.reshape((X_six.shape[0] + X_not_six.shape[0],) + shape_ord)

    # Appending the labels and converting the labels to
    # binary classification(Six=1,Not Six=0)
    Y_labels = np.append(Y_six,Y_not_six)
    Y_train = Y_labels == 6
    Y_train = Y_train.astype(int)

    # Converting the classes to its binary categorical form
    nb_classes = 2
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    # actually try to build/fit the model
    build_model(shape_ord, nb_classes, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()
