import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pylab
# plt.rcParams['figure.figsize'] = 10, 10

import os

#Import Keras.
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

#define our model
def getModel(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2):
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3), input_shape=(75, 75, 3)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    # gmodel.add(BatchNormalization())
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(drop_out))

    #Dense Layer 2
    gmodel.add(Dense(256))
    # gmodel.add(BatchNormalization())
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(drop_out))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lr_decay)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    # es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    tboard = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    # return [es, msave]
    return [msave, tboard]

# file_path = ".model_weights.hdf5"
# callbacks = get_callbacks(filepath=file_path, patience=5)

def standardize(feature):
    '''
    subtract the mean of feature from feature then divide by variance
    '''
    temp = feature.copy()
    return (feature - temp.mean()) / (temp.std()**2 * 1.0)

def data_pipeline(raw_data):
    #Generate the training data
    #Create 3 bands having HH, HV and avg of both
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_1"]])
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_2"]])

    # make a channel that is the average of the two channels
    X_avg_band = (X_band_1+X_band_2)/2.0

    # TODO: need to normalize data at some point
    X_band_1 = standardize(X_band_1)
    X_band_2 = standardize(X_band_2)
    X_avg_band = standardize(X_avg_band)

    data = np.concatenate([X_band_1[..., np.newaxis], X_band_2[..., np.newaxis], X_avg_band[..., np.newaxis]], axis=-1)

    return data

def augment_data(X_train, y_train):
    #############################################
    ### Flip Left/right for data augmentation ###
    #############################################

    # don't know how many channels X_train will have but it should be the last dimension
    # so make a list comprehension using the shape?

    process_data = [X_train[...,channel]  for channel in range(X_train.shape[3])]

    # make an empty list to store intermediary arrays
    new_data = []

    # step through each channel
    for channel in process_data:
        # new channel is the vertical axis flipped data
        new_channel = np.concatenate((channel, channel[...,::-1]), axis=0)

        # add the new axis now and it will save work later
        new_data.append(new_channel[..., np.newaxis])

    double_X_train = np.concatenate(new_data, axis=-1)

    double_y_train = np.concatenate((y_train, y_train), axis=0)

    # random shuffle the arrays because currently it's first half original
    # second half mirror. This might cause some weirdness in training?
    p = np.random.permutation(double_X_train.shape[0])

    return double_X_train[p], double_y_train[p]

def plot_hist(hist, epochs, learning_rate, batch_size, drop_out, lr_decay):
    fig, axs = plt.subplots(1,2,figsize=(16, 8))

    info_str = "v1_epochs_{}_lr_{}_lrdecay_{}_batch_{}_dropout_{}.png".format(epochs,
                    learning_rate,
                    lr_decay,
                    batch_size,
                    drop_out)
    info_str = info_str.replace("1e-", "")

    fig.suptitle(info_str, fontsize=12, fontweight='normal')

    major_ticks = int(epochs/10.0)
    minor_ticks = int(epochs/20.0)
    if major_ticks < 2: major_ticks = 2
    if minor_ticks < 1: minor_ticks = 1

    majorLocator = MultipleLocator(major_ticks)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(minor_ticks)

    # correct x axis
    hist.history['loss'] = [0.0] + hist.history['loss']
    hist.history['val_loss'] = [0.0] + hist.history['val_loss']
    hist.history['acc'] = [0.0] + hist.history['acc']
    hist.history['val_acc'] = [0.0] + hist.history['val_acc']

    x_line = [0.2] * (epochs + 1)

    axs[0].set_title("Iceberg/Ship Classifier Loss Function Error\n Train Set and Dev Set")
    axs[0].set_xlabel('Epochs')
    axs[0].set_xlim(1, epochs)
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 1.5)
    axs[0].plot(x_line, color="red", alpha=0.3, lw=4.0)
    axs[0].plot(hist.history['loss'], color="blue", linestyle="--", alpha=0.8, lw=1.0)
    axs[0].plot(hist.history['val_loss'], color="blue", alpha=0.8, lw=1.0)
    axs[0].plot(x_line, color="red", linestyle="--", alpha=0.8, lw=1.0)
    axs[0].legend(["Minimum Acceptable Error", 'Training', 'Validation'])
    axs[0].xaxis.set_major_locator(majorLocator)
    axs[0].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[0].xaxis.set_minor_locator(minorLocator)


    axs[1].set_title("Iceberg/Ship Classifier Accuracy\n Train Set and Dev Set")
    axs[1].set_xlabel('Epochs')
    axs[1].set_xlim(1, epochs)
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(0.5, 1.0)
    axs[1].plot(hist.history['acc'], color="blue", linestyle="--", alpha=0.5, lw=1.0)
    axs[1].plot(hist.history['val_acc'], color="blue", alpha=0.8, lw=1.0)
    axs[1].legend(['Training', 'Validation'], loc='lower right')
    axs[1].xaxis.set_major_locator(majorLocator)
    axs[1].xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    axs[1].xaxis.set_minor_locator(minorLocator)

    plt.savefig("../imgs/" + info_str, facecolor='w', edgecolor='w', transparent=False)
    plt.show()

def score_model(gmodel, file_path, X_train, y_train, X_dev, y_dev):
    gmodel.load_weights(filepath=file_path)

    # train set scoring
    score = gmodel.evaluate(X_train, y_train)
    print("\n\nTrain Loss: {:1.4f}".format(score[0]))
    print("Train Accuracy: {:2.3f}\n".format(score[1] * 100.0))

    # dev set scoring
    score = gmodel.evaluate(X_dev, y_dev)
    print("\n\nDev Loss: {:1.4f}".format(score[0]))
    print("Dev Accuracy: {:2.3f}\n".format(score[1] * 100.0))

def main():

    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=10)

    print("Load data...")
    #Load the data.
    train = pd.read_json("../data/train.json")
    #test = pd.read_json("../data/test.json")
    print("Data loading complete")

    print("Feed raw data into data pipeline...")
    all_X_train = data_pipeline(train)
    all_Y_train = train.loc[:,'is_iceberg']
    print("Data pipeline operations should be complete")

    print("carve data into train/dev/test sets")
    # high iteration training/testing so carve out a final validation block
    # which will be scored 10 times max
    # keep the seed stable so we're not inadvertently using all of the data/overfitting
    X_train_work, X_test, y_train_work, y_test = train_test_split(all_X_train, all_Y_train, random_state=317, train_size=0.85)

    # now do the actual split for the train/dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_work, y_train_work, train_size=0.80)
    print("data carving completed")

    print("attempt to augment data")
    X_train, y_train = augment_data(X_train, y_train)
    print("data augmentation complete")

    # epochs for model
    epochs = 1000
    learning_rate = 0.0005
    lr_decay = 6e-5
    batch_size=32
    drop_out = 0.40

    print("create Keras model")
    gmodel = getModel(learning_rate=learning_rate, lr_decay=lr_decay, drop_out=drop_out)
    print("fit Keras NN")
    hist = gmodel.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_dev, y_dev),
                callbacks=callbacks)

    print("/n/n/n/nModel fit completed")

    print("plot model error/accuracy curves")
    plot_hist(hist, epochs, learning_rate, batch_size, drop_out, lr_decay)

    print("score model")
    score_model(gmodel, file_path, X_train, y_train, X_dev, y_dev)

if __name__ == "__main__":
    main()
