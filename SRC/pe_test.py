"""
Tests the MultipleTrajTracker for the analysis of a multiple-members
forecast.
"""
import numpy as np
import datetime as dt
import os
import sys
from tools import load_hdf5_images, parse_coordinates_range
from configparser import ConfigParser
from prediction import model_segmentation, resize
from correction import filter_smallest_islets
from TST import MultipleTrajTracker, validity_range

if __name__ == "__main__":
    # ======================= DATA LOADING  =========================
    cfg = ConfigParser()
    cfg.read("config_members.cfg")

    if len(sys.argv) != 2:
        print("Usage: python SRC/pe_test.py <domain>")
        sys.exit(-1)

    # Loads each member of the forecast and assembles them into an array
    # "inputs"
    input_dir = cfg.get("paths", "input_dir")
    inputs = []
    for h5file in os.listdir(input_dir):
        input_data = load_hdf5_images(os.path.join(input_dir, h5file))
        input_data = resize(input_data)
        inputs.append(input_data)
    inputs = np.stack(inputs)

    # Dates
    basis = dt.datetime(2017, 9, 17, 0)
    terms = input_data.shape[0]
    validities = validity_range(basis, terms, time_step=1)
    model_path = cfg.get("paths", "model")

    domain = sys.argv[1]
    latitudes, longitudes = parse_coordinates_range(cfg.get("domains", domain))
    # We need to reverse the latitudes array, since it actually increases
    # downwards w/ regards to the pixels
    latitudes = np.ascontiguousarray(latitudes[::-1])

    # ======================= PREDICTION ============================
    segmentations = []
    for input_data in inputs:
        segmentation = model_segmentation(input_data, model_path)
        segmentation = np.rot90(segmentation, k=1, axes=(1, 2))
        segmentation = filter_smallest_islets(segmentation)
        segmentations.append(segmentation)
    segmentations = np.stack(segmentations)

    # ======================= POST PROCESSING =======================

    # ======================= TRACKING ==============================
    tracker = MultipleTrajTracker(validities, latitudes, longitudes)
    for data, seg in zip(inputs, segmentations):
        tracker.add_trajectory(seg, np.rot90(data[:, 0], axes=(1, 2)))
    tracker.plot_traj_probabilities("pe_1h.png")
