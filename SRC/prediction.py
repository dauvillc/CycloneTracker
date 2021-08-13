"""
Defines the functions for the prediction step.
"""
import numpy as np
from tensorflow import keras

# Normalization values
min_train = [0.0014705637004226446, -0.0005646797362715006]
max_train = [84.3562137670815, 0.005240639875410125]


# While the user could load the model on their own
# using keras.models.load_model(), it's better to make them
# use a custom function so that we can make any change here
# and remain coherent with the other functions.
def load_model(model_path):
    """
    Loads and returns the prediction model.
    :param model_path: path to the model (either in SavedModel
        or H5 format).
    :return: the model to use for the other prediction functions.
    """
    return keras.models.load_model(model_path)


def resize(images):
    """
    Resizes a batch of images (Input data, target data, coastline..)
    to fit the model's output size.
    :param images: Batch of images of shape (n_imgs, channels, h, w, [...]).
        (The array must have at least 4 dimensions but may have more).
    :return: A copy of images whose bottom and right limits have been
        cropped to fit the target dimensions.
    Currently, the model's output size is the closest lower multiple of 8
    to the input dimensions.
    """
    height, width = images.shape[2], images.shape[3]
    new_h, new_w = int(height / 8) * 8, int(width / 8) * 8
    return images[:, :, :new_h, :new_w]


def make_prediction(input_data, model, batch_size=1):
    """
    Uses a model to make the predictions over a batch of input
    fields.
    :param input_data: array of shape (n_samples, 2, height, width).
        The two channels must be the FF10m and TA850 fields for each
        sample in the batch.
    :param model: Model to be used for the prediction. Should be loaded
        using load_model().
    :param batch_size: Number of images passed to the model at once.
        A lower batch size requires less GPU memory.
    :return: the predictions as an array of shape (n, 3, h, w).
        Each channel gives the probabilites for each class
        (Empty, Cyclonic Winds, Max Winds) at each pixel.
    """
    # Data normalization
    input_data = resize(input_data.copy())
    input_data[:, 0] = (input_data[:, 0] - min_train[0]) / (max_train[0] -
                                                            min_train[0])
    input_data[:, 1] = (input_data[:, 1] - min_train[1]) / (max_train[1] -
                                                            min_train[1])
    return model.predict(input_data, batch_size)


def make_segmentation(prediction):
    """
    Transforms a probabilities map predicted by a model
    into a segmentation image.
    :param prediction: array of shape (N, h, w, nb_classes),
        probabilities given by the model for each class at each pixel.
    :return: an array of shape (N, h, w) giving the decided
        class at each pixel.
    """
    return np.argmax(prediction, axis=1)


def model_segmentation(input_data, model_path, batch_size=1):
    """
    Does the whole job of loading the model, preprocessing
    the input data, and returning the predicted segmentations.
    :param input_data: array of shape (n_samples, height, width, 2).
        The two channels must be the FF10m and TA850 fields for each
        sample in the batch.
    :param model_path: path to the model to use, in tensorflow SavedModel
        format, or keras h5 format.
    :param batch_size: Number of images passed to the model at once.
        A lower batch size requires less GPU memory.
    :return: an array of shape (N, h, w) giving the decided class at each
        pixel. The classes are represented with indexes:
        - 0 for the empty class;
        - 1 for Cyclonic Winds (VCyc);
        - 2 for Max Winds (VMax).
    """
    model = load_model(model_path)
    prediction = make_prediction(input_data, model, batch_size)
    return make_segmentation(prediction)
