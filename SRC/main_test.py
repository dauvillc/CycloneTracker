"""
File to execute in order to start the cyclone tracker.
"""
import numpy as np
import datetime as dt
import sys
from configparser import ConfigParser
from prediction import model_segmentation
from correction import filter_smallest_islets
from TST import SingleTrajTracker, validity_range, load_single_traj_tracker
from epygram_data import load_data_from_grib

if __name__ == "__main__":
    # ======================= DATA LOADING  =========================
    cfg = ConfigParser()
    cfg.read("config_tracker.cfg")
    # Dates
    basis = dt.datetime(2021, 9, 27, 12)
    terms = 78
    validities = validity_range(basis, terms, time_step=1)
    model_path = cfg.get("paths", "model")

    # Geography
    domain = sys.argv[1]
    input_data = []
    latitudes, longitudes = None, None
    for val in validities:
        data, latitudes, longitudes = load_data_from_grib(
            basis, val.term(), domain)
        input_data.append(data)
    input_data = np.stack(input_data)

    # ======================= PREDICTION ============================
    segmentation = model_segmentation(input_data, model_path)
    segmentation = np.rot90(segmentation, k=1, axes=(1, 2))
    """
    path = "/home/dauvilliersc/meteofrancerepo/preds/unet_OI_corrected.obj"
    with open(path, "rb") as obj:
        segmentation = np.rot90(load(obj)[:, 0], k=1, axes=(1, 2))
    """

    # ======================= POST PROCESSING =======================
    segmentation = filter_smallest_islets(segmentation)

    # ======================= TRACKING ==============================
    tracker = SingleTrajTracker(latitudes, longitudes)
    for mask, val, ff10 in zip(segmentation, validities, input_data):
        tracker.add_new_state(mask, val, np.rot90(ff10[0]))
    tracker.plot_current_trajectory("traj.png")
    tracker.evolution_graph("evolution.png")
    tracker.save("saved_tracker")

    tracker = load_single_traj_tracker("saved_tracker")
    for mask, val, ff10 in zip(segmentation, validities, input_data):
        tracker.add_new_state(mask, val, np.rot90(ff10[0]))
    tracker.plot_current_trajectory("traj.png")
    tracker.evolution_graph("evolution.png")
