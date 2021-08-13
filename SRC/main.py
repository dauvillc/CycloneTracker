"""
File to execute in order to start the cyclone tracker.
"""
import os
import numpy as np
import datetime as dt
from epygram.base import FieldValidity
from tools import parse_coordinates_range
from configparser import ConfigParser
from prediction import resize, load_model, make_prediction, make_segmentation
from correction import filter_smallest_islets
from TST import SingleTrajTracker
from epygram_data import load_data_from_grib
from paramiko import SSHClient
from scp import SCPClient

if __name__ == "__main__":
    # ======================= DATA LOADING  =========================
    cfg = ConfigParser()
    cfg.read("config_tracker.cfg")

    latitudes = parse_coordinates_range(cfg.get("geography", "latitudes"))
    # Reverses the latitudes array, since the latitude increases upwards
    # whereas the indexes of pixels increase downwards
    latitudes = np.ascontiguousarray(latitudes[::-1])
    longitudes = parse_coordinates_range(cfg.get("geography", "longitudes"))
    domain = cfg.get("geography", "domain")
    save_dir = cfg.get("paths", "save_directory")

    model_path = cfg.get("paths", "model")
    model = load_model(model_path)

    # ======================= SUCCESSIVE TRACKING ===================

    # The basis will span along a given range of dates
    # All basis are those at 0h
    # For each basis, we'll segment and track the objects for the
    # first 12 terms (from +0h to +11h)

    # Original basis
    day = dt.datetime(2021, 8, 13, 0)
    term = dt.timedelta(hours=50)
    last_day = dt.datetime(2021, 8, 13)
    tracker = SingleTrajTracker(latitudes, longitudes)
    while day <= last_day:
        # Dates
        basis = day
        # If we're at term +12h, we switch to the next day instead
        if term.total_seconds() == 3600 * 79:
            term = dt.timedelta(seconds=0)
            day += dt.timedelta(days=1)
        # Assemble the basis and term into a validity
        validity = FieldValidity(basis + term, basis=basis, term=term)

        # Load the input data from Vortex
        input_data = np.expand_dims(load_data_from_grib(basis, term, domain),
                                    axis=0)
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

    # Creates a directory and saves the results into it
    tmp_save_dir = day.strftime("tracking_%Y%m%d%H")
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)
    tracker.plot_current_trajectory(
        os.path.join(tmp_save_dir, "ended_trajectories.png"))
    tracker.plot_trajectories(
        os.path.join(tmp_save_dir, "ongoing_trajectory.png"))

    # Creates an SSH connexion to copy the results to the save dir
    with SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.connect("sxcoope1")

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(tmp_save_dir, recursive=True, remote_path=save_dir)
