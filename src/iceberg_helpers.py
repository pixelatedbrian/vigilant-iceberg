import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def train_test_dev_split(vars, seed1=2017, seed2=2018, test_size=0.15, dev_size=0.25):
    '''
    Like SK-Learn's train_test_split but uses a randomly shuffled index to correctly
    split 3 or more items instead of just 2 for train_test_split

    Parameters:
    vars:       tuple of (X_images, inc_angle, y_labels)
    seed1:      seed for the split that pulls the test set out
    seed2:      seed for the split that does normal train/dev split
    test_size:  what ratio of data to pull out for test set
    dev_size:   what ratio f data to pull out for dev set

    Returns:
    Dictionary with:
        X_images_train, X_images_dev, X_images_test
        inc_angle_train, inc_angle_dev, inc_angle_test
        y_train, y_dev, y_test
    '''

    _len = vars[0].shape[0]

    indices = np.random.RandomState(seed1).permutation(_len)

    test_index = int(_len * (1 - test_size))
    test_indices = indices[test_index:]

    remaining_indices = indices[:test_index]

    dev_index = int(remaining_indices.shape[0] * (1 - dev_size))
    dev_indices = remaining_indices[dev_index:]

    train_indices = remaining_indices[:dev_index]

    data = {}
    data["X_images_train"] = vars[0][train_indices]
    data["inc_angle_train"] = vars[1][train_indices]
    data["y_train"] = vars[2][train_indices]

    data["X_images_dev"] = vars[0][dev_indices]
    data["inc_angle_dev"] = vars[1][dev_indices]
    data["y_dev"] = vars[2][dev_indices]

    data["X_images_test"] = vars[0][test_indices]
    data["inc_angle_test"] = vars[1][test_indices]
    data["y_test"] = vars[2][test_indices]

    return data


def standardize(feature, standard_params=None):
    '''
    subtract the mean of feature from feature then divide by variance

    if standard_params is not None then we're fitting test to the train
    mean and std. Use the values being passed in as a tuple of mean and std
    looking like:
        (mean, std)
    '''
    if standard_params is not None:
        return (feature - standard_params[0] / (standard_params[1] * 1.0)), standard_params
    else:
        temp = feature.copy()
        _mean = temp.mean()
        _std = temp.std()

        return (feature - _mean) / (_std * 1.0), (_mean, _std)


def data_augmentation(imgs, ud=False, rotate90=False):
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


def blur_images(imgs, perc_chop=60):
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


def data_pipeline(raw_data, standardization_params=None, special_c3=False):
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

    if standardization_params is not None:
        # Done: need to normalize data at some point
        X_band_1, s_params_c0 = standardize(X_band_1, standardization_params["c0"])
        X_band_2, s_params_c1 = standardize(X_band_2, standardization_params["c1"])
        X_band_3, s_params_c2 = standardize(X_band_3, standardization_params["c2"])

    else:
        # Done: need to normalize data at some point
        X_band_1, s_params_c0 = standardize(X_band_1)
        X_band_2, s_params_c1 = standardize(X_band_2)
        X_band_3, s_params_c2 = standardize(X_band_3)

    data = np.concatenate([X_band_1[..., np.newaxis],
                           X_band_2[..., np.newaxis],
                           X_band_3[..., np.newaxis]],
                          axis=-1)

    standardized_params = {"c0": s_params_c0,
                           "c1": s_params_c1,
                           "c2": s_params_c2}

    return data, standardized_params


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
    # plt.show()


def score_model(gmodel, file_path, X_train, y_train, X_dev, y_dev):
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


def new_score_model(gmodel, file_path, data_dict, score_test=False):
    '''
    Takes in a keras model and scores it with the provided data and labels
    Prints out the results of the scoring

    Parameters:
    gmodel:     keras (tensorflow) model that will predict on the provided data
    file_path:  where to find the model weights to load
    data_dict:  dictionary of data with keys as follows:
                # keys: "X_images_train", "inc_angle_train", "y_train"
                #       "X_images_dev", "inc_angle_dev", y_dev"
                #       "X_images_test", "inc_angle_test", y_test"

    Returns:
    float of the score of the dev set, or else None
    '''
    gmodel.load_weights(filepath=file_path)

    _set = "_train"
    # train set scoring
    score = gmodel.evaluate([data_dict["X_images" + _set], data_dict["inc_angle" + _set]],
                            data_dict["y" + _set])

    print("\n\nTrain Loss: {:1.4f}".format(score[0]))
    print("Train Accuracy: {:2.3f}\n".format(score[1] * 100.0))

    # dev set scoring
    _set = "_dev"
    score = gmodel.evaluate([data_dict["X_images" + _set], data_dict["inc_angle" + _set]],
                            data_dict["y" + _set])

    print("\n\nDev Loss: {:1.4f}".format(score[0]))
    print("Dev Accuracy: {:2.3f}\n".format(score[1] * 100.0))

    if score_test is True or score[0] < 0.18:
        _set = "_test"

        score = gmodel.evaluate([data_dict["X_images" + _set], data_dict["inc_angle" + _set]],
                                data_dict["y" + _set])

        print("\n\nTest Loss: {:1.4f}".format(score[0]))
        print("Test Accuracy: {:2.3f}\n".format(score[1] * 100.0))

        return score[0]

    return None
