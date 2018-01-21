import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from iceberg_helpers import plot_hist, data_pipeline, train_test_dev_split, standardize, new_score_model, data_augmentation
from model_zoo import gmodel, gmodel2

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


# from keras import initializers
import time


def main():
    _path = "../data/.indigo_1499_model_weights"
    _model = gmodel2()
    _model.load_weights(filepath=_path)

    mypotim = Adam(lr=0.001,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-08,
                   decay=1e-6)

    _model.compile(loss='binary_crossentropy',
                   optimizer=mypotim,
                   metrics=['accuracy'])

    print("Load data...")
    # Load the data.
    train = pd.read_json("../data/train.json")
    # test = pd.read_json("../data/test.json")
    print("Data loading complete")

    print("Feed raw data into data pipeline...")
    all_X_pics, standardized_params = data_pipeline(train, special_c3=False)
    # all_Y_train = train.loc[:,'is_iceberg']

    # figure out extra X features from training data
    inc_angle = pd.to_numeric(train.loc[:, "inc_angle"], errors="coerce")

    inc_mean = inc_angle.mean()

    inc_angle[np.isnan(inc_angle)] = inc_mean
    # inc_angle = np.array(inc_angle, dtype=np.float32)

    # TODO: enable this?
    # inc_angle, inc_std_params = standardize(inc_angle)

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

    print("score train set:")
    score = _model.evaluate([data_dict["X_images_train"], data_dict["inc_angle_train"]], data_dict["y_train"], verbose=1)
    print("\n")
    print('Train score: {:2.2f}'.format(score[0]))
    print('Train accuracy: {:2.2f}%'.format(score[1] * 100.0))

    print("score dev set:")
    score = _model.evaluate([data_dict["X_images_dev"], data_dict["inc_angle_dev"]], data_dict["y_dev"], verbose=1)
    print("\n")
    print('Dev score: {:2.2f}'.format(score[0]))
    print('Dev accuracy: {:2.2f}%'.format(score[1] * 100.0))

    print("score test set:")
    score = _model.evaluate([data_dict["X_images_test"], data_dict["inc_angle_test"]], data_dict["y_test"], verbose=1)
    print("\n")
    print('Test score: {:2.2f}'.format(score[0]))
    print('Test accuracy: {:2.2f}%'.format(score[1] * 100.0))

    df_test = pd.read_json('../data/test.json')

    print("Push through data pipeline...", standardized_params)
    test_pics, _ = data_pipeline(df_test, standardized_params, special_c3=False)

    # print("pipeline inc_angle", inc_std_params)
    test_inc_angle = pd.to_numeric(df_test.loc[:, "inc_angle"], errors="coerce")

    # fill the holes with the mean of the previous mean from training
    test_inc_angle[np.isnan(test_inc_angle)] = inc_mean
    # inc_angle = np.array(inc_angle, dtype=np.float32)

    # TODO: enable this?
    # has the (mean, std) from standardizing inc_angle earlier
    # test_inc_angle, _ = standardize(test_inc_angle, inc_std_params)

    print("test_inc_angle", test_inc_angle[:10])

    # because there used to be a column for inc_angle isnan, but that seems
    # to cause issues as that only occurs in training data but not test data

    print("Pipeline complete")
    print("\n")
    print("Predict on transformed data")

    # df_test.inc_angle = df_test.inc_angle.replace('na',0)
    # Xtest = (get_scaled_imgs(df_test))
    # Xinc = df_test.inc_angle
    # pred_test = model.predict([Xtest,Xinc])

    # pred_test = _model.predict([data_dict["X_images_train"], data_dict["inc_angle_train"]], verbose=1)
    pred_test = _model.predict([test_pics, test_inc_angle], verbose=1)

    # print("preds: ", pred_test[:10])

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))


if __name__ == "__main__":
    main()
