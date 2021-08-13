"""
File to execute in order to start the cyclone tracker.
"""
import numpy as np
import datetime as dt
from epygram.base import FieldValidity
from tools import parse_coordinates_range
from configparser import ConfigParser
from prediction import resize, load_model, make_prediction, make_segmentation
from correction import filter_smallest_islets
from TST import SingleTrajTracker
from epygram_data import load_data_from_grib

if __name__ == "__main__":
    # ======================= DATA LOADING  =========================
    cfg = ConfigParser()
    cfg.read("config_tracker.cfg")

    latitudes = parse_coordinates_range(cfg.get("geography", "latitudes"))
    # Reverses the latitudes array, since the latitude increases upwards
    # whereas the indexes of pixels increase downwards
    latitudes = np.ascontiguousarray(latitudes[::-1])
    longitudes = parse_coordinates_range(cfg.get("geography", "longitudes"))

    model_path = cfg.get("paths", "model")
    model = load_model(model_path)

    # ======================= SUCCESSIVE TRACKING ===================

    # The basis will span along a given range of dates
    # All basis are those at 0h
    # For each basis, we'll segment and track the objects for the
    # first 12 terms (from +0h to +11h)

    # Original basis
    day = dt.datetime(2020, 3, 16)
    term = dt.timedelta(hours=0)
    last_day = dt.datetime(2020, 3, 20)
    tracker = SingleTrajTracker(latitudes, longitudes)
    while day <= last_day:
        # Dates
        basis = day
        # If we're at term +12h, we switch to the next day instead
        if term.total_seconds() == 3600 * 12:
            term = dt.timedelta(seconds=0)
            day += dt.timedelta(days=1)
        # Assemble the basis and term into a validity
        validity = FieldValidity(basis + term, basis=basis, term=term)

        # Load the input data from Vortex
        input_data = np.expand_dims(load_data_from_grib(basis, term), axis=0)
        input_data = resize(input_data)
        # Add 1 hour to the term for the next loop iteration
        term += dt.timedelta(hours=1)

        # Make the prediction / segmentation and correct it
        probas = make_prediction(input_data, model, batch_size=2)
        segmentation = make_segmentation(probas)
        segmentation = np.rot90(segmentation, k=1, axes=(1, 2))
        segmentation = filter_smallest_islets(segmentation)

        # Track
        tracker.add_new_state(segmentation[0], validity,
                              np.rot90(input_data[0, 0]))
    tracker.plot_current_trajectory("results/exemple_automatized.png")
    tracker.plot_trajectories("results/automatic_detection.png")
