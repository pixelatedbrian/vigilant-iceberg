import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from iceberg_helpers import plot_hist, score_model, data_pipeline

# import tensorflow as tf

#Import Keras.
# from keras.preprocessing.image import ImageDataGenerator
# from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
# from keras.utils import layer_utils
# from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
# from keras import regularizers

from keras.layers.merge import concatenate
# from keras import initializers


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
    # X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((3, 3), strides=(2, 2), name="max_pool0")(X)

    # DROPOUT
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 1 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(128, (3, 3), strides=(1, 1), name="conv1")(X)
    # X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool1")(X)

    # DROPOUT
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

    # DROPOUT
    X = Dropout(drop_out)(X)

    ##############
    ### CONV 3 ###
    ##############

    # CONV -> BN -> RELU
    X = Conv2D(512, (3, 3), strides=(1, 1), name="conv3")(X)
    # X = BatchNormalization(axis=3, name='bn_conv3')(X)
    X = Activation("relu")(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=(2, 2), name="max_pool3")(X)

    # DROPOUT
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

    return model


#define our model
def getModel(learning_rate=0.001, lr_decay=1e-6, drop_out=0.2, input_shape=(75, 75, 3)):
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(7, 7), input_shape=input_shape))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(5, 5)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 3
    gmodel.add(Conv2D(256, kernel_size=(3, 3)))
    # gmodel.add(BatchNormalization(axis=3))
    gmodel.add(Activation('relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(drop_out))

    #Conv Layer 4
    gmodel.add(Conv2D(512, kernel_size=(3, 3)))
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

    mypotim = Adam(lr=learning_rate,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-08,
                   decay=lr_decay)

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


def augment_data(X_train_images, X_train_non_images, y_train):
    #############################################
    ### Flip Left/right for data augmentation ###
    #############################################

    # don't know how many channels X_train will have but it should be the last dimension
    # so make a list comprehension using the shape?

    process_data = [X_train_images[..., channel] for channel in range(X_train_images.shape[3])]

    # make an empty list to store intermediary arrays
    new_data = []

    # step through each channel
    for channel in process_data:
        # new channel is the vertical axis flipped data
        new_channel = np.concatenate((channel, channel[..., ::-1]), axis=0)

        # flip vertically
        # new_channel = np.concatenate((new_channel, channel[:,::-1,:]), axis=0)
        #
        # # flip vertically and horizontally
        # new_channel = np.concatenate((new_channel, channel[:,::-1,::-1]), axis=0)

        # add the new axis now and it will save work later
        new_data.append(new_channel[..., np.newaxis])

    double_X_train_images = np.concatenate(new_data, axis=-1)

    double_X_train_nonimages = np.concatenate((X_train_non_images, X_train_non_images), axis=0)

    double_y_train = np.concatenate((y_train, y_train), axis=0)

    # random shuffle the arrays because currently it's first half original
    # second half mirror. This might cause some weirdness in training?
    p = np.random.permutation(double_X_train_images.shape[0])

    return double_X_train_images[p], double_X_train_nonimages[p], double_y_train[p]


def main():

    file_path = ".model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=10)

    print("Load data...")
    # Load the data.
    train = pd.read_json("../data/train.json")
    # test = pd.read_json("../data/test.json")
    print("Data loading complete")

    print("Feed raw data into data pipeline...")
    all_X_pics = data_pipeline(train)
    # all_Y_train = train.loc[:,'is_iceberg']

    # figure out extra X features from training data
    inc_angle = pd.to_numeric(train.loc[:, "inc_angle"], errors="coerce")
    inc_angle[np.isnan(inc_angle)] = inc_angle.mean()
    # inc_angle = np.array(inc_angle, dtype=np.float32)

    # inc_angle = standardize(inc_angle)

    # print("inc_angle type: ", type(inc_angle))
    # #
    # inc_angle = tf.convert_to_tensor(inc_angle, np.float32)

    # because there used to be a column for inc_angle isnan, but that seems
    # to cause issues as that only occurs in training data but not test data
    all_X_nonpics = inc_angle

    # Get labels
    all_Y_labels = train.loc[:, "is_iceberg"]

    # make X data linspace so that we can use train_test_split on that and then use
    # indices to get the slices that we need
    x_indices = np.arange(all_X_pics.shape[0])

    print("all x pics shape:", all_X_pics.shape)
    print("all x nonpics shape:", all_X_nonpics.shape)
    print("shape of x_indices:", x_indices.shape)
    print("shape of y labels:", all_Y_labels.shape)

    print("Data pipeline operations should be complete")

    print("carve data into train/dev/test sets")
    # high iteration training/testing so carve out a final validation block
    # which will be scored 10 times max
    # keep the seed stable so we're not inadvertently using all of the data/overfitting
    X_train_work_indices, X_test_indices, y_train_work, y_test = train_test_split(x_indices, all_Y_labels, random_state=317, train_size=0.85)

    # figure out what the train slices are
    # these slices are work in progress as they will be sliced again
    X_train_work_pics = all_X_pics[X_train_work_indices]
    X_train_work_nonpics = all_X_nonpics[X_train_work_indices]

    # figure out the test holdout slices
    X_test_pics = all_X_pics[X_test_indices]
    X_test_nonpics = all_X_nonpics[X_test_indices]

    # make new linspace to get sliced
    x_indices = np.arange(X_train_work_pics.shape[0])

    # now do the actual split for the train/dev sets
    X_train_indices, X_dev_indices, y_train, y_dev = train_test_split(x_indices, y_train_work, train_size=0.80, random_state=12018)

    X_train_pics = X_train_work_pics[X_train_indices]
    X_train_nonpics = X_train_work_nonpics[X_train_indices]

    X_dev_pics = X_train_work_pics[X_dev_indices]
    X_dev_nonpics = X_train_work_nonpics[X_dev_indices]

    print("X_train_images shape:", X_train_pics.shape)
    print("X_train_non_images shape:", X_train_nonpics.shape)
    print("y_Train shape:", y_train.shape)

    print("data carving completed")

    print("attempt to augment data")
    X_train_pics, X_train_nonpics, y_train = augment_data(X_train_pics, X_train_nonpics, y_train)
    print("data augmentation complete")

    # epochs for model
    epochs = 50
    learning_rate = 0.001
    lr_decay = 1e-6
    batch_size = 32
    drop_out = 0.20

    print("create Keras model")
    # icy_model = tiny_icy_model((75, 75, 3), drop_out)
    _model = gmodel2(learning_rate, lr_decay, drop_out)

    mypotim = Adam(lr=learning_rate,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-08,
                   decay=lr_decay)

    _model.compile(loss='binary_crossentropy',
                   optimizer=mypotim,
                   metrics=['accuracy'])

    # optimo = Adam(lr=learning_rate, decay=lr_decay)
    #
    # _model.compile(optimizer=optimo, loss="binary_crossentropy", metrics=["accuracy"])

    _model.summary()

    # gmodel = getModel(learning_rate=learning_rate, lr_decay=lr_decay, drop_out=drop_out)
    print("fit Keras NN")

    # hist = icy_model.fit([X_train_pics, X_train_nonpics], y_train,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             verbose=1,
    #             validation_data=([X_dev_pics, X_dev_nonpics], y_dev),
    #             callbacks=callbacks)

    # hist = gmodel.fit(X_train, y_train,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             verbose=1,
    #             validation_data=(X_dev, y_dev),
    #             callbacks=callbacks)

    hist = _model.fit([X_train_pics, X_train_nonpics], y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=([X_dev_pics, X_dev_nonpics], y_dev),
                      callbacks=callbacks)

    print("\n\n\nModel fit completed")

    print("plot model error/accuracy curves")
    plot_hist(hist, epochs, learning_rate, batch_size, drop_out, lr_decay)

    print("score model")
    score_model(_model, file_path, [X_train_pics, X_train_nonpics], y_train,
                                   [X_dev_pics, X_dev_nonpics], y_dev)

if __name__ == "__main__":
    main()
    # print "hi"
