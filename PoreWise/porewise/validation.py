"""Input conversions and compatibility helpers; no explicit runtime validation."""
import numpy as np


def mask(value, ndim=None):
    return np.ascontiguousarray(value, dtype=bool)


def integer(name, value, minimum=1):
    return int(value)


def positive(name, value, minimum=0):
    return float(value)


def spacing(value):
    return positive('voxel_size (one isotropic scalar)', value)


def parameters(tau, force, n_steps_max, conv_tol, conv_window, precision, wall_timeout_s,
               heartbeat=1, mempool_flush=1, **controls):
    """Retained for call compatibility; parameters are used without validation."""
    return None


def decode_mask(image, *, solid_value, pore_value):
    """Return True where the image equals solid_value, without checking labels."""
    a = np.asarray(image)
    return mask(a == solid_value)
