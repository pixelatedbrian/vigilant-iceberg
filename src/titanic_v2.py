# Evolve iceberg_helpers into a full class
import numpy as np
import cv2
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from model_zoo_v2 import model1, model2, model3, model4, tiny_model, model5
from scipy import signal  # for "get_worms" feature transformation

from sklearn.model_selection import train_test_split

import time

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)


class Titanic(object):
    '''
    Version 2

    Skip three level split since we're being greedy for data.

    Drop inc_angle with nan values

    Make data channels only 2 'colors' instead of 3 with a nonsensical 3rd channel

    Don't include inc_angle in model data
    '''

    def __init__(self, model_name="model2",
                 lr=0.001,
                 lr_decay=1e-6,
                 drop_out=0.45,
                 batch_size=32,
                 epochs=100,
                 augment_rotate=False,
                 augment_ud=True,
                 c3_transform=0):

        # paths to the data
        self.train_path = "../data/train.json"
        self.test_path = "../data/test.json"

        # this will hold a dictionary that contains the means and standard
        # deviations of all items sent through the pipeline
        # these can then be reused when we need to predict through this model
        self.standardization_params = {}

        # how much to multiply single column data after image augmentation
        # for example images x 4 then data_amp = 4 so y_train * 4
        self.data_amp = 0

        self.data_dict = None

        self.inc_mean = 0

        # model hyperparameters
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.drop_out = drop_out
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment_rotate = augment_rotate
        self.augment_ud = augment_ud
        self.c3_transform = c3_transform

        self.img_shape = (75, 75, 2)

        self.model = None
        self.model_name = model_name
        self.model_dict = {"model1": model1,
                           "model2": model2,
                           "model3": model3,
                           "model4": model4,
                           "tiny_model": tiny_model,
                           "model5": model5}

        self.callbacks = None

        self.file_path = "indigo_model_weights.hdf5"

        # loss of the model on dev set
        self.dev_score = 0

    def run_me(self):
        '''
        Does all of the steps to run the model in the first place

        Troubleshooting:
            Disable
        '''

        print("Data Pipeline: Begin")
        # load the training data
        print("Data Pipeline: > Load data...")
        train_data = pd.read_json(self.train_path)
        print("Data Pipeline: >> Data loading complete")

        # standardize the data
        print("Data Pipeline: >>> Standardizing image training data...")
        # TODO: make special_c3 a class veriable so we don't have to care about
        # it when predicting training data too
        all_X_pics = self.data_pipeline(train_data)

        print("Data Pipeline: >>>> Load y labels...")
        all_Y_labels = train_data.loc[:, "is_iceberg"]
        print("Data Pipeline: >>>>> Load y labels complete")

        # figure out extra X features from training data
        print("Data Pipeline: >>>>>> Load inc_angle training data...")
        train_data.loc[:, "inc_angle"] = train_data.loc[:, "inc_angle"].replace('na', 0)

        # find non-nan indicies
        nan_idx = np.where(train_data.loc[:, "inc_angle"] > 0)

        print("Data Pipeline: >>>>>>> Drop nan inc_angle rows")

        all_Y_labels = all_Y_labels[nan_idx[0]]
        all_X_pics = all_X_pics[nan_idx[0], ...]

        print("all_Y_labels shape", all_Y_labels.shape)
        print("all_X_labels.shape", all_X_pics.shape)

        time.sleep(5)

        print("Data Pipeline: >>>>>>>> Data standardizing complete")

        # split into train/test
        print("Data Pipeline: >>>>>>>>> Split into train/test sets...")
        # self.train_test_split(all_X_pics, all_Y_labels)
        self.data_dict = {}
        self.data_dict["X_train"], self.data_dict["X_test"], self.data_dict["y_train"], self.data_dict["y_test"] = train_test_split(all_X_pics, all_Y_labels)

        print("Data Pipeline: >>>>>>>>> Augment data")
        self.data_dict["X_train"], self.data_amp = self.data_augmentation(self.data_dict["X_train"])
        self.data_dict["y_train"] = self.amplify_data(self.data_dict["y_train"])

        print("Data Pipeline: > Augmentation complete")

        print("Data Pipeline: >> Shuffle augmented data")
        self.data_dict["X_train"], self.data_dict["y_train"] = self.shuffle_training_data(self.data_dict["X_train"], self.data_dict["y_train"])

        print("Data Pipeline: >>> Data carving complete.")
        print("Data Pipeline: >>>> Complete")

        print("Model: > Instantiate Model")
        self.model = self.model_dict[self.model_name](self.learning_rate,
                                                      self.lr_decay,
                                                      self.drop_out,
                                                      self.img_shape)

        print("Model: >> Establish Callbacks")
        self.callbacks = self.get_callbacks()

        print("Model: >>> Fit Model")
        hist = self.model.fit(self.data_dict["X_train"], self.data_dict["y_train"],
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              verbose=1,
                              validation_data=(self.data_dict["X_test"], self.data_dict["y_test"]),
                              callbacks=self.callbacks)

        print("\n\n\nModel fit completed")

        print("plot model error/accuracy curves")
        self.plot_hist(hist)

        print("score model")
        train_loss, test_loss = self.new_score_model()

        print("score of test set:", test_loss)

        # try to log data so we can keep track what is going on
        with open("scores.txt", "a") as outfile:
            out = "{:0.5f}~{:0.5f}~{:0.6f}~{:0.8f}~{:0.6f}~{:03d}\n".format(test_loss,
                                                                            train_loss,
                                                                            self.learning_rate,
                                                                            self.lr_decay,
                                                                            self.drop_out,
                                                                            self.batch_size)
            outfile.write(out)

        if test_loss <= 0.175:

            print("Save Model Weights with dev score:")
            score_text = ("{:0.3f}_".format(test_loss)).replace(".", "_")
            self.model.save_weights(score_text + self.file_path)
            # load and score the test set
            self.predict_test_set(score_text)

    def predict_test_set(self, score_text):
        '''
        does the pipeline stuff for the test set and then runs predict on
        that data
        '''
        test_data = pd.read_json(self.test_path)

        print("Data Pipeline: >>> Standardizing image test data...")
        test_X_pics = self.data_pipeline(test_data)

        # figure out extra X features from training data
        print("Data Pipeline: >>>>> Load inc_angle training data...")
        test_data.loc[:, "inc_angle"] = test_data.loc[:, "inc_angle"].replace('na', 0)

        # find non-nan indicies
        nan_idx = np.where(test_data.loc[:, "inc_angle"] > 0)

        print("Data Pipeline: >>>>>> Drop nan inc_angle rows")

        test_X_pics = test_X_pics[nan_idx[0], ...]

        print("Data Pipeline: >>>>>> Data standardizing complete")

        # load just completed weights
        self.model.load_weights(filepath=self.file_path)

        pred_test = self.model.predict(test_X_pics)

        # print("prediction data:", data[:10])

        submission = pd.DataFrame({'id': test_data["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

        print(submission.head(10))

        # add score to csv
        submission.to_csv(score_text + 'cnn.csv', index=False)

    def score_model(self, gmodel, file_path, X_train, y_train, X_dev, y_dev):
        '''
        Takes in a keras model and scores it with the provided data and labels
        Prints out the results of the scoring

        Parameters:
        gmodel:     keras (tensorflow) model that will predict on the provided data
        file_path:  where to find the model weights to load
        X_train:    data trained on
        y_train:    labels for training set
        X_dev:      data for evaluation
        y_dev:      labels for evaluation

        Returns:
        None
        '''
        gmodel.load_weights(filepath=file_path)

        # train set scoring
        score = gmodel.evaluate(X_train, y_train)
        print("\n\nTrain Loss: {:1.4f}".format(score[0]))
        print("Train Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        # dev set scoring
        score = gmodel.evaluate(X_dev, y_dev)
        print("\n\nDev Loss: {:1.4f}".format(score[0]))
        print("Dev Accuracy: {:2.3f}\n".format(score[1] * 100.0))

    def new_score_model(self, score_test=False):
        '''
        Takes in a keras model and scores it with the provided data and labels
        Prints out the results of the scoring

        Parameters:
        gmodel:     keras (tensorflow) model that will predict on the provided data
        file_path:  where to find the model weights to load
        train_data_dict:  dictionary of data with keys as follows:
                    # keys: "X_images_train", "inc_angle_train", "y_train"
                    #       "X_images_dev", "inc_angle_dev", y_dev"
                    #       "X_images_test", "inc_angle_test", y_test"

        Returns:
        float of the score of the dev set, or else None
        '''
        self.model.load_weights(filepath=self.file_path)

        _set = "_train"
        # train set scoring
        score = self.model.evaluate(self.data_dict["X" + _set],
                                    self.data_dict["y" + _set])

        print("\n\nTrain Loss: {:1.4f}".format(score[0]))
        print("Train Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        train_loss = score[0]

        # dev set scoring
        _set = "_test"
        score = self.model.evaluate(self.data_dict["X" + _set],
                                    self.data_dict["y" + _set])

        print("\n\nDev Loss: {:1.4f}".format(score[0]))
        print("Dev Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        test_loss = score[0]

        return train_loss, test_loss

    def plot_hist(self, hist):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        info_str = "m2_c1_epochs_{:03d}_lr_{:0.5f}_lrdecay_{:0.8f}_batch_{:03d}_dropout_{:0.5f}.png".format(self.epochs,
                                                                                                            self.learning_rate,
                                                                                                            self.lr_decay,
                                                                                                            self.batch_size,
                                                                                                            self.drop_out)
        info_str = info_str.replace("1e-", "")

        fig.suptitle(info_str, fontsize=12, fontweight='normal')

        major_ticks = int(self.epochs / 10.0)
        minor_ticks = int(self.epochs / 20.0)

        if major_ticks < 2:
            major_ticks = 2

        if minor_ticks < 1:
            minor_ticks = 1

        majorLocator = MultipleLocator(major_ticks)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(minor_ticks)

        # correct x axis
        hist.history['loss'] = [0.0] + hist.history['loss']
        hist.history['val_loss'] = [0.0] + hist.history['val_loss']
        hist.history['acc'] = [0.0] + hist.history['acc']
        hist.history['val_acc'] = [0.0] + hist.history['val_acc']

        x_line = [0.2] * (self.epochs + 1)

        axs[0].set_title("Iceberg/Ship Classifier Loss Function Error\n Train Set and Dev Set")
        axs[0].set_xlabel('Epochs')
        axs[0].set_xlim(1, self.epochs)
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
        axs[1].set_xlim(1, self.epochs)
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
        # plt.show()

    def get_callbacks(self):
        # es = EarlyStopping('val_loss', patience=patience, mode="min")

        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(self.file_path, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
        # tboard = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # return [es, msave]
        # return [msave, tboard]
        return [earlyStopping, mcp_save, reduce_lr_loss]

    def shuffle_training_data(self, X, y):
        '''
        See if randomizing the order helps with training because otherwise it
        would be the same pic 8, 32, whatever times in a row which probably
        would have an impact on model learning.
        '''
        # random shuffle the arrays because currently it's first half original
        # second half mirror. This might cause some weirdness in training?
        p = np.random.permutation(X.shape[0])

        print("shuffle augmented data")
        # now shuffly augmented data:
        return X[p], y[p]

    def amplify_data(self, data):
            '''
            Basically multiply whatever data and return it again
            '''
            temp = [data for idx in range(self.data_amp)]

            return np.concatenate(temp)

    def data_augmentation(self, imgs):
        #############################################
        # Flip Left/right for data augmentation #####
        #############################################

        # keep track of how much the data is augmented
        # so inc_angle and labels can also be augmented properly
        channel_mult = 2

        # don't know how many channels X_train will have but it should be the last dimension
        # so make a list comprehension using the shape?
        process_data = [imgs[..., channel] for channel in range(imgs.shape[3])]

        # make an empty list to store intermediary arrays
        new_data = []

        if self.augment_ud is True:
            channel_mult += 2

        # step through each channel
        for channel in process_data:
            # Always flips L/R axis.  Too tired to engineer more flexible solution
            # this method will consider if it should flip up/down though

            # new channel is the vertical axis flipped data
            new_channel = np.concatenate((channel, channel[..., ::-1]), axis=0)

            if self.augment_ud is True:
                # flip vertically
                new_channel = np.concatenate((new_channel, channel[:, ::-1, :]), axis=0)

                # flip vertically and horizontally
                new_channel = np.concatenate((new_channel, channel[:, ::-1, ::-1]), axis=0)

            # add the new axis now and it will save work later
            new_data.append(new_channel[..., np.newaxis])

        double_imgs = np.concatenate(new_data, axis=-1)

        # ok, try to rotate each image 90 degrees
        if self.augment_rotate is True:
            channel_mult *= 8

            # already have 1 orientation so need to add 3 more dirs
            for ibx in range(3):
                # rotation transform matrix for OpenCV
                M = cv2.getRotationMatrix2D((75 / 2, 75 / 2), 90 + 90 * ibx, 1)
                res = np.array([cv2.warpAffine(img, M, (75, 75)) for img in double_imgs])

                double_imgs = np.concatenate([double_imgs, res])

        print("Channel_mult", channel_mult)

        return double_imgs, channel_mult

    def train_test_split(self, X, y, seed1=1337, test_size=0.15):
            '''
            Like SK-Learn's train_test_split but uses a randomly shuffled index to correctly
            split 3 or more items instead of just 2 for train_test_split

            Parameters:
            X:          images of all data
            y:          labels of all data
            seed1:      seed for the split that pulls the test set out
            test_size:  what ratio of data to pull out for test set

            Creates self.data_dict:
                X_train, y_train, X_test, y_test

            Returns:
                None
            '''

            print("y[:50]", y[np.isnan(y)])

            _len = X.shape[0]

            # make a shuffled list of indices
            indices = np.random.RandomState(seed1).permutation(_len)

            # figure out which indices correspond to the test set
            test_index = int(_len * (1 - test_size))
            test_indices = indices[test_index:]

            # gather the leftover indices
            train_indices = indices[:test_index]

            self.data_dict = {}

            self.data_dict["X_train"] = X[train_indices]
            self.data_dict["y_train"] = y[train_indices]

            self.data_dict["X_test"] = X[test_indices]
            self.data_dict["y_test"] = y[test_indices]

            print("y_train[:50]", self.data_dict["y_train"][np.isnan(self.data_dict["y_train"])])

            # print(self.data_dict["X_train"].shape)
            # print(self.data_dict["y_train"].shape)
            # print(self.data_dict["X_test"].shape)
            # print(self.data_dict["y_test"].shape)

    def standardize(self, feature_data, _key):
        '''
        subtract the mean of feature from feature then divide by variance

        if standard_params is not None then we're fitting test to the train
        mean and std. Use the values being passed in as a tuple of mean and std
        looking like:
            (mean, std)
        '''

        # see if the key is in the parameter dictionary keys. If it is then
        # load data needed from that. Otherwise add key/value pair to dict
        if _key in self.standardization_params.keys():

            temp_mean = self.standardization_params[_key][0]
            temp_std = self.standardization_params[_key][1]

            return (feature_data - temp_mean) / (temp_std * 1.0)
        else:
            # make a shallow copy so that we don't mess up values when we work
            # on the arrays
            temp = feature_data.copy()
            _mean = temp.mean()
            _std = temp.std()

            # add key/value pair to dictionary
            self.standardization_params[_key] = (_mean, _std)

            return (feature_data - _mean) / (_std * 1.0)

    def scaler(self, feature_data, _key):
        '''
        subtract the mean of feature from feature then divide by the difference
        of the max value minus the min value. (Normalize to range of 0.0 to 1.0)

        if standard_params is not None then we're fitting test to the train
        mean and std. Use the values being passed in as a tuple of mean and std
        looking like:
            (mean, std)
        '''

        # see if the key is in the parameter dictionary keys. If it is then
        # load data needed from that. Otherwise add key/value pair to dict
        if _key in self.standardization_params.keys():

            temp_mean = self.standardization_params[_key][0]
            temp_maxmin = self.standardization_params[_key][1]

            return (feature_data - temp_mean) / temp_maxmin
        else:
            # make a shallow copy so that we don't mess up values when we work
            # on the arrays
            temp = feature_data.copy()
            _mean = temp.mean()
            _maxmin = temp.max() - temp.min()

            # add key/value pair to dictionary
            self.standardization_params[_key] = (_mean, _maxmin)

            return (feature_data - _mean) / _maxmin

    def data_pipeline(self, raw_data):
        # Generate the training data
        # Create 3 bands having HH, HV and avg of both
        # Modified for v2 to only have 2 channels
        #
        # TODO: if doing special transformations then do it to both X_1 and X_2
        #       and not to an extra channel.  This on the theory that adding a
        #       weird transformation will make the data that the filters are
        #       looking at inconsistent. If you do 'worms' then do it to all,
        #       not just some.  Not sure if this is a good idea but worth trying

        X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_1"]])
        X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_2"]])

        if self.standardization_params is not None:
            # Done: need to normalize data at some point
            X_band_1 = self.standardize(X_band_1, "c0")
            X_band_2 = self.standardize(X_band_2, "c1")

            # img_shape is (75, 75, 3) normally but could in theory
            # get 4 or 5 channels
            # TODO: figure out how this would work, may not have to do anything

            # X_band_1 = self.scaler(X_band_1, "c0")
            # X_band_2 = self.scaler(X_band_2, "c1")
            # X_band_3 = self.scaler(X_band_3, "c2")

        else:
            # Done: need to normalize data at some point
            X_band_1 = self.standardize(X_band_1, "c0")
            X_band_2 = self.standardize(X_band_2, "c1")

        # restack the channels into one matrix 3 'colors' deep
        data = np.concatenate([X_band_1[..., np.newaxis],
                               X_band_2[..., np.newaxis]],
                              axis=-1)

        return data

    def get_worms(self, data):
        xderivative = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        yderivative = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        arrx = signal.convolve2d(data, xderivative, mode="valid")
        arry = signal.convolve2d(data, yderivative, mode="valid")
        worms = np.hypot(arrx, arry)

        worms = np.lib.pad(worms, ((1, 1), (1, 1)), "mean")

        return worms

    def blur_images(self, imgs, perc_chop=60):
        '''
        Takes in an array of images and modifies them with blur filtering

        returns an array of images with the same shape as passed in
        '''
        # upscale to give the blur algs more room to work
        temp = np.array([cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for img in imgs])

        # percentage to chop the bottom out at:
        _percentile = np.percentile(temp, perc_chop)

        print("Image Blurring: Percentile setting {:} value: {:}".format(perc_chop, _percentile))

        # cut out the lowest levels of the image since it (appears) to be all
        # noise.  (We'll see if that is a correct intuition!)
        temp[temp < _percentile] = _percentile

        # apply median blur
        blurred = np.array([cv2.medianBlur(blur, 5) for blur in temp])

        # apply a bigger filter to blur
        blurred = np.array([cv2.bilateralFilter(te, 25, 45, 45) for te in blurred])

        # shrink back down to previous size
        blurred = np.array([cv2.resize(blu, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) for blu in blurred])

        return blurred
