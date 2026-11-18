"""Utilities for removing void components that are isolated from the boundary."""

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage


def fill_isolated_voids(binary: ArrayLike) -> np.ndarray:
    """Return a copy with void components not connected to a volume face filled.

    The input must be a three-dimensional binary image using ``0`` for void
    and ``1`` for solid. Connectivity is face-sharing (6-connectivity).
    """
    image = np.asarray(binary)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D binary image, got {image.ndim} dimensions")
    if any(size == 0 for size in image.shape):
        raise ValueError("binary image dimensions must be non-empty")
    if not np.all((image == 0) | (image == 1)):
        raise ValueError("binary image must contain only 0 (void) and 1 (solid)")

    void_labels, component_count = ndimage.label(
        image == 0, structure=ndimage.generate_binary_structure(3, 1)
    )
    if component_count == 0:
        return np.array(image, dtype=np.uint8, copy=True)

    boundary_labels = np.unique(
        np.concatenate(
            (
                void_labels[0, :, :].ravel(),
                void_labels[-1, :, :].ravel(),
                void_labels[:, 0, :].ravel(),
                void_labels[:, -1, :].ravel(),
                void_labels[:, :, 0].ravel(),
                void_labels[:, :, -1].ravel(),
            )
        )
    )
    connected_to_boundary = np.zeros(component_count + 1, dtype=bool)
    connected_to_boundary[boundary_labels] = True
    isolated_voids = (void_labels != 0) & ~connected_to_boundary[void_labels]

    filled = np.array(image, dtype=np.uint8, copy=True)
    filled[isolated_voids] = 1
    return filled


__all__ = ["fill_isolated_voids"]
