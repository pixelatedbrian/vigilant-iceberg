import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from iceberg_helpers import plot_hist, data_pipeline, train_test_dev_split, standardize, new_score_model, data_augmentation

# import tensorflow as tf

# Import Keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras import regularizers

from keras.layers.merge import concatenate
# from keras import initializers
import time


def get_more_images(imgs):

    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def gmodel2(learning_rate=0.001, drop_out=0.2, input_shape=(75, 75, 3)):
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

    # DROPOUT
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
    X = BatchNormalization(axis=3, name='bn_conv3')(X)
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


def get_callbacks(filepath, patience=2):
    # es = EarlyStopping('val_loss', patience=patience, mode="min")

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    # tboard = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    # return [es, msave]
    # return [msave, tboard]
    return [earlyStopping, mcp_save, reduce_lr_loss]


def amplify_data(data, multiplier):
    '''
    Basically multiply whatever data and return it again
    '''
    temp = [data for idx in range(multiplier)]

    return np.concatenate(temp)


def main():
    _run = True

    count = 0

    while _run is True:
        try:

            count += 1

            file_path = ".{:}_indigo_model_weights.hdf5".format(count)
            callbacks = get_callbacks(filepath=file_path)

            print("Load data...")
            # Load the data.
            train = pd.read_json("../data/train.json")
            # test = pd.read_json("../data/test.json")
            print("Data loading complete")

            print("Feed raw data into data pipeline...")
            all_X_pics, standardized_params = data_pipeline(train)
            # all_Y_train = train.loc[:,'is_iceberg']

            # figure out extra X features from training data
            inc_angle = pd.to_numeric(train.loc[:, "inc_angle"], errors="coerce")
            inc_angle[np.isnan(inc_angle)] = inc_angle.mean()
            # inc_angle = np.array(inc_angle, dtype=np.float32)

            # TODO: enable this?
            inc_angle, inc_std_params = standardize(inc_angle)

            # because there used to be a column for inc_angle isnan, but that seems
            # to cause issues as that only occurs in training data but not test data

            # Get labels
            all_Y_labels = train.loc[:, "is_iceberg"]

            print("Data pipeline operations should be complete")

            print("carve data into train/dev/test sets")
            # high iteration training/testing so carve out a final validation block
            # which will be scored 10 times max
            # keep the seed stable so we're not inadvertently using all of the data/overfitting

            # keys: "X_images_train", "inc_angle_train", "y_train"
            #       "X_images_dev", "inc_angle_dev", y_dev"
            #       "X_images_test", "inc_angle_test", y_test"
            data_dict = train_test_dev_split((all_X_pics, inc_angle, all_Y_labels))

            print("X_images_train shape:", data_dict["X_images_train"].shape)
            print("inc_angle_train shape:", data_dict["inc_angle_train"].shape)
            print("y_train shape:", data_dict["y_train"].shape)

            print("X_images_dev shape:", data_dict["X_images_dev"].shape)
            print("inc_angle_dev shape:", data_dict["inc_angle_dev"].shape)
            print("y_dev shape:", data_dict["y_dev"].shape)

            print("X_images_test shape:", data_dict["X_images_test"].shape)
            print("inc_angle_test shape:", data_dict["inc_angle_test"].shape)
            print("y_test shape:", data_dict["y_test"].shape)

            print("data carving completed")

            print("attempt to augment data")
            # X_train_pics, X_train_nonpics, y_train = augment_data(X_train_pics, X_train_nonpics, y_train)
            data_dict["X_images_train"], data_amp = data_augmentation(data_dict["X_images_train"], ud=True, rotate90=True)
            data_dict["inc_angle_train"] = amplify_data(data_dict["inc_angle_train"], data_amp)
            data_dict["y_train"] = amplify_data(data_dict["y_train"], data_amp)

            # random shuffle the arrays because currently it's first half original
            # second half mirror. This might cause some weirdness in training?
            p = np.random.permutation(data_dict["X_images_train"].shape[0])

            print("shuffle augmented data")
            # now shuffly augmented data:
            data_dict["X_images_train"][p]
            data_dict["inc_angle_train"][p]
            data_dict["y_train"][p]
            # return double_X_train_images[p], double_X_train_nonimages[p], double_y_train[p]

            print("data augmentation complete")

            # epochs for model
            epochs = 100

            # aiming for ~0.001 - 0.0001
            _exp = (np.random.uniform(-5.5, -3.0))
            learning_rate = 4**_exp

            # aiming for ~0.0001 - 0.000001
            _exp = (np.random.uniform(-8.5, -6.5))
            lr_decay = 4.0**_exp
            # learning_rate = 0.0001
            # lr_decay = 5e-6

            batches = [16, 32, 48, 64, 96, 128]
            batch_size = batches[np.random.randint(0, len(batches) - 1)]

            drop_out = np.random.uniform(0.05, 0.6)
            # batch_size = 32
            # drop_out = 0.275

            print("create Keras model")
            # icy_model = tiny_icy_model((75, 75, 3), drop_out)
            _model = gmodel2(learning_rate, drop_out)

            mypotim = Adam(lr=learning_rate,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-08,
                           decay=lr_decay)

            _model.compile(loss='binary_crossentropy',
                           optimizer=mypotim,
                           metrics=['accuracy'])

            # _model.summary()

            print("fit Keras NN")
            time.sleep(5.0)
            print("Launching ~ ~ ~ >>-----<>")

            hist = _model.fit([data_dict["X_images_train"], data_dict["inc_angle_train"]], data_dict["y_train"],
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=([data_dict["X_images_dev"], data_dict["inc_angle_dev"]], data_dict["y_dev"]),
                              callbacks=callbacks)

            print("\n\n\nModel fit completed")

            print("plot model error/accuracy curves")
            plot_hist(hist, epochs, learning_rate, batch_size, drop_out, lr_decay)

            print("score model")
            score_test = new_score_model(_model, file_path, data_dict)

            if score_test is not None:
                df_test = pd.read_json('../data/test.json')
                test_pics, _ = data_pipeline(df_test, standardized_params)

                test_inc_angle = pd.to_numeric(df_test.loc[:, "inc_angle"], errors="coerce")
                test_inc_angle[np.isnan(test_inc_angle)] = test_inc_angle.mean()
                # inc_angle = np.array(inc_angle, dtype=np.float32)

                # TODO: enable this?
                # has the (mean, std) from standardizing inc_angle earlier
                test_inc_angle, _ = standardize(test_inc_angle, inc_std_params)

                # because there used to be a column for inc_angle isnan, but that seems
                # to cause issues as that only occurs in training data but not test data

                pred_test = _model.predict([test_pics, test_inc_angle])

                submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
                print(submission.head(10))

                file_name = '{:1.4f}_cnn.csv'.format(score_test)
                submission.to_csv(file_name, index=False)

        except ValueError:
            print(ValueError)

if __name__ == "__main__":
    main()
    # print "hi"
