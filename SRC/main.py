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
from copy import deepcopy
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
    # Disable Tensorflow's logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

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

    # Original basis
    if len(sys.argv) != 2:
        print("Usage: python main.py <basis hours>")
        sys.exit(-1)
    basis_hours = int(sys.argv[1])

    # Retrieves yesterday's basis
    initial_day = dt.datetime.combine(dt.date.today(),
                                      dt.time(hour=basis_hours))
    initial_day -= dt.timedelta(days=1)

    # ======================= SUCCESSIVE TRACKING ===================

    # The basis will span along a given range of dates
    # All basis are those at 0h
    # For each basis, we'll segment and track the objects for the
    # first 12 terms (from +0h to +11h)

    day = deepcopy(initial_day)
    term = dt.timedelta(hours=0)
    last_day = day
    basis = initial_day
    tracker = SingleTrajTracker(latitudes, longitudes)
    while day <= last_day:
        # We try to load the grib for basis and term until we reach
        # a term that doesn't exist (usually +49h or +73h)
        try:
            print("Fetching GRIB for {}+{:1f}h".format(
                day,
                term.total_seconds() / 3600))
            # Assemble the basis and term into a validity
            validity = FieldValidity(basis + term, basis=basis, term=term)

            # Load the input data from Vortex
            input_data = np.expand_dims(load_data_from_grib(
                basis, term, domain),
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
            # Add 1h to the term for the next iteration
            term += dt.timedelta(hours=1)
        except IOError:
            # CASE: we didn't find the GRIB for this term, we switch
            # to the next day
            day += dt.timedelta(days=1)
            term = dt.timedelta()

    # Creates a directory and saves the results into it
    tmp_save_dir = initial_day.strftime("%Y-%m-%d-%H")
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)
    tracker.plot_current_trajectory(
        os.path.join(tmp_save_dir, "ongoing_trajectory.png"))
    tracker.plot_trajectories(
        os.path.join(tmp_save_dir, "ended_trajectories.png"))
    tracker.evolution_graph(os.path.join(tmp_save_dir, "evolution_chart.png"))

    # Saves the tracker's data
    print("Saving the tracker..")
    tracker.save(cfg.get("paths", "tracker_save_dir"))

    # Destroys the "shouldfly-" files created by vortex as they are HEAVY
    os.system("rm -rf shouldfly-*")

    # Creates an SSH connexion to copy the results to the save dir
    # The destination directory must contains a folder "domain" for the
    # domain that is being treated
    print("Copying results to", os.path.join(save_dir, tmp_save_dir))
    with SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.connect("sxcoope1")

        with SCPClient(ssh.get_transport()) as scp:
            remote_path = os.path.join(save_dir, domain)
            scp.put(tmp_save_dir, recursive=True, remote_path=remote_path)
