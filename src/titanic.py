# Evolve iceberg_helpers into a full class
import numpy as np
import cv2
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from model_zoo import gmodel, gmodel2

class Titanic(object):

    def __init__(self, model_name="gmodel2",
                       lr=0.001,
                       lr_decay=1e-6,
                       drop_out=0.45,
                       batch_size=64,
                       epochs=100):

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

        self.train_data_dict = None

        self.inc_mean = 0

        # model hyperparameters
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.drop_out = drop_out
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None

        self.callbacks = None

        self.file_path = "indigo_model_weights.hdf5"



    def run_me(self):
        '''
        Does all of the steps to run the model in the first place
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
        all_X_pics = self.data_pipeline(train_data, special_c3=False)

        # figure out extra X features from training data
        print("Data Pipeline: >>>> Load inc_angle training data...")
        inc_angle = pd.to_numeric(train_data.loc[:, "inc_angle"], errors="coerce")

        self.inc_mean = inc_angle.mean()

        print("Data Pipeline: >>>>> Standardizing inc_angle training data...")
        inc_angle[np.isnan(inc_angle)] = self.inc_mean
        # inc_angle = np.array(inc_angle, dtype=np.float32)

        # TODO: enable this?
        inc_angle = self.standardize(inc_angle, "inc_angle")

        print("Data Pipeline: >>>>>> Data standardizing complete")

        print("Data Pipeline: >>>>>>> Load y labels...")
        all_Y_labels = train_data.loc[:, "is_iceberg"]
        print("Data Pipeline: >>>>>>>> Load y labels complete")

        # split into train/dev/test
        print("Data Pipeline: >>>>>>>>> Split into train/dev/test sets...")
        self.train_dev_test_split((all_X_pics, inc_angle, all_Y_labels))

        print("Data Pipeline: >>>>>>>>>> Data carving complete.")

        print("Data Pipeline: > Augment data")
        # X_train_pics, X_train_nonpics, y_train = augment_data(X_train_pics, X_train_nonpics, y_train)
        self.train_data_dict["X_images_train"], self.data_amp = self.data_augmentation(self.train_data_dict["X_images_train"], ud=True, rotate90=True)
        self.train_data_dict["inc_angle_train"] = self.amplify_data(self.train_data_dict["inc_angle_train"])
        self.train_data_dict["y_train"] = self.amplify_data(self.train_data_dict["y_train"])

        print("Data Pipeline: >> Shuffle augmented data")
        self.shuffle_training_data()

        print("Data Pipeline: >>> Augmentation complete")
        print("Data Pipeline: >>>> Complete")

        print("Model: > Instantiate Model")

        # TODO: take in model string and select which model to use
        self.model = gmodel(self.learning_rate, self.lr_decay, self.drop_out)

        self.callbacks = self.get_callbacks()

        hist = self.model.fit([self.train_data_dict["X_images_train"], self.train_data_dict["inc_angle_train"]], self.train_data_dict["y_train"],
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          verbose=1,
                          validation_data=([self.train_data_dict["X_images_dev"], self.train_data_dict["inc_angle_dev"]], self.train_data_dict["y_dev"]),
                          callbacks=self.callbacks)

        print("\n\n\nModel fit completed")

        print("plot model error/accuracy curves")
        self.plot_hist(hist)

        print("score model")
        score_test = self.new_score_model()

        print("score of test set:", score_test)

        # load and score the test set
        self.predict_test_set()


    def predict_test_set(self):
        '''
        does the pipeline stuff for the test set and then runs predict on
        that data
        '''
        test_data = pd.read_json(self.test_path)

        print("Data Pipeline: >>> Standardizing image test data...")
        test_X_pics = self.data_pipeline(test_data, special_c3=False)

        # figure out extra X features from training data
        print("Data Pipeline: >>>> Load inc_angle training data...")
        test_inc_angle = pd.to_numeric(test_data.loc[:, "inc_angle"], errors="coerce")

        print("Data Pipeline: >>>>> Standardizing inc_angle from training data mean...")
        test_inc_angle[np.isnan(test_inc_angle)] = self.inc_mean
        # inc_angle = np.array(inc_angle, dtype=np.float32)

        # TODO: enable this?
        test_inc_angle = self.standardize(test_inc_angle, "inc_angle")

        print("Data Pipeline: >>>>>> Data standardizing complete")

        # load just completed weights
        self.model.load_weights(filepath=self.file_path)

        pred_test = self.model.predict([test_X_pics, test_inc_angle])

        # print("prediction data:", data[:10])

        submission = pd.DataFrame({'id': test_data["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
        
        print(submission.head(10))

        submission.to_csv('cnn.csv', index=False)


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
        score = self.model.evaluate([self.train_data_dict["X_images" + _set], self.train_data_dict["inc_angle" + _set]],
                                     self.train_data_dict["y" + _set])

        print("\n\nTrain Loss: {:1.4f}".format(score[0]))
        print("Train Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        # dev set scoring
        _set = "_dev"
        score = self.model.evaluate([self.train_data_dict["X_images" + _set], self.train_data_dict["inc_angle" + _set]],
                                     self.train_data_dict["y" + _set])

        print("\n\nDev Loss: {:1.4f}".format(score[0]))
        print("Dev Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        if score_test is True or score[0] < 0.18:
            _set = "_test"

            score = self.model.evaluate([self.train_data_dict["X_images" + _set], self.train_data_dict["inc_angle" + _set]],
                                         self.train_data_dict["y" + _set])

            print("\n\nTest Loss: {:1.4f}".format(score[0]))
            print("Test Accuracy: {:2.3f}\n".format(score[1] * 100.0))

            return score[0]

        return None


    def plot_hist(self, hist):
        fig, axs = plt.subplots(1,2,figsize=(16, 8))

        info_str = "v1_epochs_{}_lr_{}_lrdecay_{}_batch_{}_dropout_{}.png".format(self.epochs,
                        self.learning_rate,
                        self.lr_decay,
                        self.batch_size,
                        self.drop_out)
        info_str = info_str.replace("1e-", "")

        fig.suptitle(info_str, fontsize=12, fontweight='normal')

        major_ticks = int(self.epochs/10.0)
        minor_ticks = int(self.epochs/20.0)
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

        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(self.file_path, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
        # tboard = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # return [es, msave]
        # return [msave, tboard]
        return [earlyStopping, mcp_save, reduce_lr_loss]

    def shuffle_training_data(self):
        '''
        See if randomizing the order helps with training because otherwise it
        would be the same pic 8, 32, whatever times in a row which probably
        would have an impact on model learning.
        '''
        # random shuffle the arrays because currently it's first half original
        # second half mirror. This might cause some weirdness in training?
        p = np.random.permutation(self.train_data_dict["X_images_train"].shape[0])

        print("shuffle augmented data")
        # now shuffly augmented data:
        self.train_data_dict["X_images_train"][p]
        self.train_data_dict["inc_angle_train"][p]
        self.train_data_dict["y_train"][p]


    def amplify_data(self, data):
            '''
            Basically multiply whatever data and return it again
            '''
            temp = [data for idx in range(self.data_amp)]

            return np.concatenate(temp)


    def data_augmentation(self, imgs, ud=False, rotate90=False):
        #############################################
        ### Flip Left/right for data augmentation ###
        #############################################

        # keep track of how much the data is augmented
        # so inc_angle and labels can also be augmented properly
        channel_mult = 2

        # don't know how many channels X_train will have but it should be the last dimension
        # so make a list comprehension using the shape?
        process_data = [imgs[..., channel] for channel in range(imgs.shape[3])]

        # make an empty list to store intermediary arrays
        new_data = []

        if ud is True:
            channel_mult += 2

        # step through each channel
        for channel in process_data:
            # Always flips L/R axis.  Too tired to engineer more flexible solution
            # this method will consider if it should flip up/down though

            # new channel is the vertical axis flipped data
            new_channel = np.concatenate((channel, channel[..., ::-1]), axis=0)

            if ud is True:
                # flip vertically
                new_channel = np.concatenate((new_channel, channel[:, ::-1, :]), axis=0)

                # flip vertically and horizontally
                new_channel = np.concatenate((new_channel, channel[:, ::-1, ::-1]), axis=0)

            # add the new axis now and it will save work later
            new_data.append(new_channel[..., np.newaxis])

        double_imgs = np.concatenate(new_data, axis=-1)

        # ok, try to rotate each image 90 degrees
        if rotate90 is True:
            channel_mult *= 8

            # already have 1 orientation so need to add 3 more dirs
            for ibx in range(3):
                # rotation transform matrix for OpenCV
                M = cv2.getRotationMatrix2D((75 / 2, 75 / 2), 90 + 90 * ibx, 1)
                res = np.array([cv2.warpAffine(img, M, (75, 75)) for img in double_imgs])

                double_imgs = np.concatenate([double_imgs, res])

        print("Channel_mult", channel_mult)

        return double_imgs, channel_mult


    def train_dev_test_split(self, _vars, seed1=2017, seed2=2018, test_size=0.15, dev_size=0.25):
            '''
            Like SK-Learn's train_test_split but uses a randomly shuffled index to correctly
            split 3 or more items instead of just 2 for train_test_split

            Parameters:
            vars:       tuple of (X_images, inc_angle, y_labels)
            seed1:      seed for the split that pulls the test set out
            seed2:      seed for the split that does normal train/dev split
            test_size:  what ratio of data to pull out for test set
            dev_size:   what ratio f data to pull out for dev set

            Creates dictionary with:
                        X_images_train, X_images_dev, X_images_test
                        inc_angle_train, inc_angle_dev, inc_angle_test
                        y_train, y_dev, y_test
            Returns:
            None
            '''

            _len = _vars[0].shape[0]

            # make a shuffled list of indices
            indices = np.random.RandomState(seed1).permutation(_len)

            # figure out which indices correspond to the test set
            test_index = int(_len * (1 - test_size))
            test_indices = indices[test_index:]

            # gather the leftover indices
            remaining_indices = indices[:test_index]

            # figure out which indices belong in the dev set
            dev_index = int(remaining_indices.shape[0] * (1 - dev_size))
            dev_indices = remaining_indices[dev_index:]

            # finally gather the training set indices
            train_indices = remaining_indices[:dev_index]

            self.train_data_dict = {}
            self.train_data_dict["X_images_train"] = _vars[0][train_indices]
            self.train_data_dict["inc_angle_train"] = _vars[1][train_indices]
            self.train_data_dict["y_train"] = _vars[2][train_indices]

            self.train_data_dict["X_images_dev"] = _vars[0][dev_indices]
            self.train_data_dict["inc_angle_dev"] = _vars[1][dev_indices]
            self.train_data_dict["y_dev"] = _vars[2][dev_indices]

            self.train_data_dict["X_images_test"] = _vars[0][test_indices]
            self.train_data_dict["inc_angle_test"] = _vars[1][test_indices]
            self.train_data_dict["y_test"] = _vars[2][test_indices]

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

            return (feature_data - temp_mean / (temp_std * 1.0))
        else:
            # make a shallow copy so that we don't mess up values when we work
            # on the arrays
            temp = feature_data.copy()
            _mean = temp.mean()
            _std = temp.std()

            # add key/value pair to dictionary
            self.standardization_params[_key] = (_mean, _std)

            return (feature_data - _mean) / (_std * 1.0)

    def data_pipeline(self, raw_data, special_c3=False):
        # Generate the training data
        # Create 3 bands having HH, HV and avg of both
        X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_1"]])
        X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in raw_data["band_2"]])

        if special_c3 is False:
            # make a channel that is the average of the two channels
            X_band_3 = (X_band_1 + X_band_2) / 2.0
        else:
            # create blurred images of the 3rd channel
            X_band_3 = blur_images((X_band_1 + X_band_2) / 2.0)

        if self.standardization_params is not None:
            # Done: need to normalize data at some point
            X_band_1 = self.standardize(X_band_1, "c0")
            X_band_2 = self.standardize(X_band_2, "c1")
            X_band_3 = self.standardize(X_band_3, "c2")

        else:
            # Done: need to normalize data at some point
            X_band_1 = self.standardize(X_band_1, "c0")
            X_band_2 = self.standardize(X_band_2, "c1")
            X_band_3 = standardize(X_band_3, "c2")

        # restack the channels into one matrix 3 'colors' deep
        data = np.concatenate([X_band_1[..., np.newaxis],
                               X_band_2[..., np.newaxis],
                               X_band_3[..., np.newaxis]],
                              axis=-1)

        return data

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
