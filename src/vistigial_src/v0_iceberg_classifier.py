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
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


#define our model
def getModel(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2):
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(drop_out))

    #Dense Layer 2
    gmodel.add(Dense(256))
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
    msave = ModelCheckpoint(filepath, save_best_only=False)
    # tboard = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    # return [es, msave]
    # return [msave, tboard]
    return [msave]

# file_path = ".model_weights.hdf5"
# callbacks = get_callbacks(filepath=file_path, patience=5)


def main():

    file_path = ".v0_model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=10)

    print("Load data...")
    # Load the data.
    train = pd.read_json("../data/train.json")
    # test = pd.read_json("../data/test.json")
    print("Data loading complete")

    print("Feed raw data into data pipeline...")
    all_X_train = data_pipeline(train)
    all_Y_train = train.loc[:, 'is_iceberg']
    print("Data pipeline operations should be complete")

    print("carve data into train/dev/test sets")
    # high iteration training/testing so carve out a final validation block
    # which will be scored 10 times max
    # keep the seed stable so we're not inadvertently using all of the data/overfitting
    X_train_work, X_test, y_train_work, y_test = train_test_split(all_X_train, all_Y_train, random_state=317, train_size=0.75)

    # now do the actual split for the train/dev sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_work, y_train_work, train_size=0.80)
    print("data carving completed")

    # epochs for model
    epochs = 100
    learning_rate = 0.0005
    lr_decay = 0.000005
    batch_size = 64
    drop_out = 0.45

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
