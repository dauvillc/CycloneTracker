"""
Use: python SRC/test_models.py [--filter]
A configuration file "config_test.cfg" should exist in the working
directory, with the following content:

[paths]
model_path = /path/to/model/
input_path = /path/to/input/data/h5/file.h5
target_path = /path/to/target/data.h5 or nothing

[domains]
# Indicate below which domain the test data covers
domain = antilles
# For each domain, indicate the latitudes and longitudes ranges
# as min_lat:max_lat:step_min_long:max_long:step
indien = -25.9:-7.5:0.025_32.75:67.6:0.025
antilles = 9.7:22.925:0.025_-75.3:-51.675:0.025

If no target data is found, only the input images and model's output
will be considered.

The input and target data must be saved in the h5 / hdf5 format, as
an object containing a single dataset "image", and of shape
(number_of_samples, channels, height, width). This implies that
the target data has shape (N, 1, H, W).

The model should be saved in the Tensorflow SavedModel format. Note
that if this system was to be changed, the load_model() from the TST
package would need to be adapted.

WATCHOUT: The model was trained on data rotated by 90 degrees
clockwise relatively to the geographical truth. Therefore, the input
data should be rotated as such. After the model has made its predictions,
the images are rotated by to their real orientations before they are
plotted.

"""

import os
import numpy as np
import matplotlib
import sys
from time import time
from configparser import ConfigParser
from datetime import datetime
from TST.tools import load_hdf5_images, parse_coordinates_range
from TST.tools import save_hdf5_images
from prediction import resize, load_model, make_prediction, make_segmentation
from TST.plot import ModelTestPlotter
from correction import filter_smallest_islets

matplotlib.use("Agg")

if __name__ == "__main__":
    # Disable Tensorflow's logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    # Load configuration paths
    config = ConfigParser()
    config.read("config_test.cfg")
    model_path = config.get("paths", "model_path")
    data_path = config.get("paths", "input_path")
    target_path = config.get("paths", "target_path")

    results_dir = "test_results" + datetime.now().strftime("%d%m_%Hh%M")

    # Geographical coordinates
    domain = config.get("domains", "domain")
    latitudes, longitudes = parse_coordinates_range(
        config.get("domains", domain))
    # We need to reverse the latitudes array, since it actually increases
    # downwards w/ regards to the pixels
    latitudes = np.ascontiguousarray(latitudes[::-1])

    if "--help" in sys.argv:
        print("Usage: python test_models.py [--filter]")

    # ======================== DATA LOADING ============================= #
    input_data = load_hdf5_images(data_path)
    print("Found input data of shape", input_data.shape)

    # Try to load the target data
    target_data = None
    if os.path.exists(target_path):
        target_data = load_hdf5_images(target_path)

    # ======================== PREPROCESSING ============================ #
    # Resizes the data to the model's output dimensions
    input_data = resize(input_data)
    if target_data is not None:
        target_data = resize(target_data)

    # ======================== PREDICTION =============================== #
    curr_time = time()
    print("Loading model")
    model = load_model(model_path)
    print("Loaded model in {:1.2f}s\nMaking predictions..".format(time() -
                                                                  curr_time))
    curr_time = time()

    # Makes the predictions, then transforms them into segmentations
    # every step is timed to give an idea of the performances
    predictions = make_prediction(input_data, model)
    masks = make_segmentation(predictions)
    print("Made predictions in {:1.2f}s".format(time() - curr_time))

    # Filter the predictions if the user requested it
    if "--filter" in sys.argv:
        print("Filtering results...")
        masks = filter_smallest_islets(masks)
        print("Filtered in {:1.2f}s".format(time() - curr_time))

    # Creating the results directory
    os.makedirs(results_dir)
    # Saves the predictions in the h5 format, in an array of shape
    # (N, 1, H, W). This is done here because the predictions' shape
    # will be changed afterwards, whereas here it's still the same
    # as the input and target data.
    save_hdf5_images(masks, os.path.join(results_dir, "predictions.h5"))

    # As explained in the module docstring,
    # the input data and thus the masks are rotated by 90° clockwise
    # relatively to their geographical orientation.
    # Hence we should now rotate them back to reality before saving
    # the results.
    masks = np.rot90(masks, axes=(1, 2))
    input_data = np.rot90(input_data, axes=(2, 3))
    if target_data is not None:
        target_data = np.rot90(target_data, axes=(2, 3))

    # ======================== SAVING =================================== #
    print("Saving images...")

    # IMAGES SAVING
    original_imgs_dir = os.path.join(results_dir, "input_images")
    predictions_dir = os.path.join(results_dir, "predictions")
    os.makedirs(original_imgs_dir)
    os.makedirs(predictions_dir)
    targets_dir = os.path.join(results_dir, "targets")

    if target_data is not None:
        os.makedirs(targets_dir)
        # The target data is loaded with shape (N, 1, H, W) to be consistent
        # with the input data shape; However we'll need it with shape
        # (N, H, W) to save.
        target_data = target_data[:, 0]

    wind_fields = input_data[:, 0]
    for index, (mask, field) in enumerate(zip(masks, wind_fields)):
        plotter = ModelTestPlotter(latitudes, longitudes, field, mask)
        # Saves the input image's wind field
        original_path = os.path.join(original_imgs_dir,
                                     "input_{}.png".format(index))
        plotter.save_image(original_path, False)

        # Saves the prediction
        pred_path = os.path.join(predictions_dir,
                                 "prediction_{}.png".format(index))
        plotter.start_new_figure()
        plotter.save_image(pred_path, True)
        print("{} / {}               ".format(index + 1, masks.shape[0]),
              end="\r")

        # Saves the groundtruth segmentation if there is any
        if target_data is not None:
            plotter = ModelTestPlotter(latitudes, longitudes, field,
                                       target_data[index])
            ground_path = os.path.join(targets_dir,
                                       "target_{}.png".format(index))
            plotter.save_image(ground_path, True)
