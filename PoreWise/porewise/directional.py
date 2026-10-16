"""Three directional permeabilities and three vector velocity fields, without quality checks.

Float32 fields use array axes (z, y, x); components ux, uy, uz are lattice dx/dt.
Each load starts independently from rest and runs the requested number of steps.
"""
import json
from pathlib import Path

import numpy as np

from .field_export import write_directional_vti
from .solver import periodic
from .tensor import mirrored
from .validation import mask, spacing


def directional_runs(blocked, voxel_size, *, force_magnitude=1e-6, mirror=False, **settings):
    """Yield (driving axis, permeability in m², vector components) one load at a time."""
    blocked = mask(blocked, 3)
    dx = spacing(voxel_size)
    if mirror:
        blocked = mirrored(blocked)
    settings.update(quality_checks=False, return_fields=True, wall_timeout_s=None)
    for j, axis in enumerate('xyz'):
        force = np.zeros(3)
        force[j] = force_magnitude
        run = periodic(blocked, force, **settings)
        permeability = np.float32(run['k_lu'] * dx**2)
        field = {name: np.asarray(run[name], dtype=np.float32) for name in ('ux', 'uy', 'uz')}
        del run
        yield axis, permeability, field
        del field


def compute_directional_permeability(blocked, voxel_size, **settings):
    """Return k_x_m2, k_y_m2, k_z_m2 and velocity_fields keyed by driving axis."""
    result = {'velocity_fields': {}}
    for axis, permeability, field in directional_runs(blocked, voxel_size, **settings):
        result[f'k_{axis}_m2'] = permeability
        result['velocity_fields'][axis] = field
    return result


def save_directional_results(
    directory,
    blocked,
    voxel_size,
    *,
    microct=None,
    mirror=False,
    vti_filename='fields.vti',
    json_filename='permeability.json',
    **settings,
):
    """Write permeability.json and a single Float32 fields.vti with image + three loads.

    microct preserves the original input image labels. If omitted, the Boolean
    solid mask is saved as 0=pore, 1=solid. Mirroring applies to both image and flow.
    Return the three scalar permeabilities for summaries such as Excel export.
    """
    blocked = mask(blocked, 3)
    image = blocked if microct is None else np.asarray(microct)
    if mirror:
        blocked = mirrored(blocked)
        for axis in range(3):
            image = np.concatenate([image, np.flip(image, axis=axis)], axis=axis)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    loads = directional_runs(blocked, voxel_size, **settings)
    permeabilities = write_directional_vti(
        directory / vti_filename, image, loads, voxel_size
    )
    (directory / json_filename).write_text(
        json.dumps(permeabilities, indent=2) + '\n'
    )
    return permeabilities
