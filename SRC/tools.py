"""
Defines useful functions for the cyclone tracker.
"""
import h5py
import numpy as np


def load_hdf5_images(path):
    """
    Loads a batch of images saved in the hdf5 format.
    The HDF5 file should contain a single key named "image".
    :param path: path to the hdf5 to read.
    """
    h5file = h5py.File(path, mode="r")
    return h5file["image"][()]  # [()] transforms to np array


def save_hdf5_images(data, dest_path):
    """
    Saves a batch of images in the hdf5 format.
    :param data: batch of images to save.
    :param dest_path: file path and name into which the
        array is saved.
    """
    h5file = h5py.File(dest_path, mode="w")
    h5file.create_dataset("image", data=data, dtype=data.dtype)


def parse_coordinates_range(str_coords):
    """
    Parses geographical coordinates and returns the latitudes /
    longitudes values defining a domain.
    :param str_coords: Coordinates string in the format
        min-lat:max-lat:step_min-long:max-long:step
    :return: Two 1D arrays latitudes, longitudes. Each value in the
        arrays gives the coordinate for the associated row / column
        in the domain.
    """
    lat_str, long_str = str_coords.split("_")

    low, high, step = lat_str.split(":")
    lats_range = np.arange(float(low), float(high), float(step))
    low, high, step = long_str.split(":")
    longs_range = np.arange(float(low), float(high), float(step))
    return lats_range, longs_range
