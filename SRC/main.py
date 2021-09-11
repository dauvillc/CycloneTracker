"""
Usage: python main.py <basis hours>
Looks for and track a potential cyclone in yesterday's AROME output,
for the specified basis.
<basis hours>: Hour of the basis from which the data should be taken from.
All data is taken through vortex as GRIB files. Should be one of
[0, 6, 12, 18].

The results are saved in a specific directory and copied via SSH to the
demonstration directory.
"""
import os
import numpy as np
import datetime as dt
import sys
from epygram.base import FieldValidity
from configparser import ConfigParser
from prediction import resize, load_model, make_prediction, make_segmentation
from correction import filter_smallest_islets
from TST import SingleTrajTracker, load_single_traj_tracker
from epygram_data import load_data_from_grib
from paramiko import SSHClient
from scp import SCPClient

if __name__ == "__main__":
    # ======================= DATA LOADING  =========================
    # Disable Tensorflow's logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    # Original basis
    if len(sys.argv) != 3:
        print("Usage: python main.py <basis hours> <domain>")
        sys.exit(-1)

    cfg = ConfigParser()
    cfg.read("config_tracker.cfg")

    save_dir = cfg.get("paths", "save_directory")

    domain = sys.argv[2]

    model_path = cfg.get("paths", "model")
    model = load_model(model_path)

    basis_hours = int(sys.argv[1])
    # Retrieves yesterday's basis
    initial_day = dt.datetime.combine(dt.date.today(),
                                      dt.time(hour=basis_hours))
    initial_day -= dt.timedelta(days=1)

    # ======================= SUCCESSIVE TRACKING ===================

    term = dt.timedelta(hours=0)
    basis = initial_day

    # Will store the data across the loops
    wind_fields, segmentations, validities = [], [], []

    # This loop ends when we reach a non-existing GRIB file
    # (Most likely because we've reached the last available term)
    while True:
        # We try to load the grib for basis and term until we reach
        # a term that doesn't exist (usually +49h or +73h)
        try:
            print("Fetching GRIB for {}+{:1f}h".format(
                basis,
                term.total_seconds() / 3600))
            # Assemble the basis and term into a validity
            validity = FieldValidity(basis + term, basis=basis, term=term)

            # Load the input data from Vortex
            # The data has shape (2, H, W) (channels ff10m and ta850)
            input_data, latitudes, longitudes = load_data_from_grib(
                basis, term, domain)

            # The prediction functions expect a batch of data, i.e.
            # with shape (batch_size, channels, height, width)
            input_data = np.expand_dims(input_data, axis=0)
            # Resizes the data to make it correspond with the model's
            # output dimensions
            input_data = resize(input_data)

            # Make the prediction / segmentation and correct it
            probas = make_prediction(input_data, model, batch_size=2)
            segmentation = make_segmentation(probas)
            segmentation = np.rot90(segmentation, k=1, axes=(1, 2))
            segmentation = filter_smallest_islets(segmentation)
            # The segmentation has shape (1, H, W)
            # the first dimension comes from the batch-intended functioning
            # of the prediction functions used.

            # Store everything for the tracking thereafter
            wind_fields.append(np.rot90(input_data[0, 0]))
            segmentations.append(segmentation[0])
            validities.append(validity)

            # Add 1h to the term for the next iteration
            term += dt.timedelta(hours=1)
        except IOError:
            # CASE: we didn't find the GRIB for this term
            break

    # TRACKING
    # Tries to load the saved tracker, or creates a new one otherwise
    tracker_save_dir = cfg.get("paths", "tracker_save_dir")
    try:
        tracker = load_single_traj_tracker(tracker_save_dir)
    except (IOError, FileNotFoundError):
        tracker = SingleTrajTracker(latitudes, longitudes)

    # Successively adds the new states to the tracker
    for wind, seg, val in zip(wind_fields, segmentations, validities):
        tracker.add_new_state(seg, val, wind)

    # Creates a directory and saves the results into it
    tmp_save_dir = initial_day.strftime("%Y-%m-%d-%H")
    tmp_save_dir = os.path.join("tmp_saves", tmp_save_dir)
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)
    tracker.plot_current_trajectory(
        os.path.join(tmp_save_dir, "ongoing_trajectory.png"))
    tracker.plot_trajectories(
        os.path.join(tmp_save_dir, "ended_trajectories.png"))
    tracker.evolution_graph(os.path.join(tmp_save_dir, "evolution_chart.png"))

    # Saves the tracker's data
    print("Saving the tracker..")
    tracker.save(tracker_save_dir)

    # Destroys the "shouldfly-" files created by vortex as they are HEAVY
    os.system("rm -f shouldfly-*")

    # Creates an SSH connexion to copy the results to the save dir
    # on sxcoope1
    # The destination directory must contains a folder "domain" for the
    # domain that is being treated
    print("Copying results to", os.path.join(save_dir, tmp_save_dir))
    with SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.connect("sxcoope1")

        with SCPClient(ssh.get_transport()) as scp:
            remote_path = os.path.join(save_dir, domain)
            scp.put(tmp_save_dir, recursive=True, remote_path=remote_path)
