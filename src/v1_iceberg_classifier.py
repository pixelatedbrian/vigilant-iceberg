import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from iceberg_helpers import plot_hist, score_model, data_pipeline

#Import Keras.
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


def augment_data(X_train, y_train):
    #############################################
    ### Flip Left/right for data augmentation ###
    #############################################

    # don't know how many channels X_train will have but it should be the last dimension
    # so make a list comprehension using the shape?

    process_data = [X_train[..., channel] for channel in range(X_train.shape[3])]

    # make an empty list to store intermediary arrays
    new_data = []

    # step through each channel
    for channel in process_data:
        # new channel is the vertical axis flipped data
        new_channel = np.concatenate((channel, channel[..., ::-1]), axis=0)

        # add the new axis now and it will save work later
        new_data.append(new_channel[..., np.newaxis])

    double_X_train = np.concatenate(new_data, axis=-1)

    double_y_train = np.concatenate((y_train, y_train), axis=0)

    # random shuffle the arrays because currently it's first half original
    # second half mirror. This might cause some weirdness in training?
    p = np.random.permutation(double_X_train.shape[0])

    return double_X_train[p], double_y_train[p]


def main():
    file_path = ".v1_model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=10)

    print("Load data...")
    # Load the data.
    train = pd.read_json("../data/train.json")
    # test = pd.read_json("../data/test.json")
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
    epochs = 100
    learning_rate = 0.0005
    lr_decay = 6e-4
    batch_size = 32
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
