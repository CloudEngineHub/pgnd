import numpy as np


def get_bounding_box_bimanual():
    return np.array([[0.0, 0.6], [-0.35, 0.45 + 0.75], [-0.65, 0.05]])  # the world frame robot workspace


def get_bounding_box():
    return np.array([[0.0, 0.6], [-0.35, 0.45], [-0.65, 0.05]])  # the world frame robot workspace
