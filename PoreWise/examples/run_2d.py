"""Minimal 2D example: x/y permeability and two vector fields, with fixed-step solves.

For three x/y/z fields from a 3D image, use python -m porewise instead.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from porewise import geometry, load_image
from porewise.solver import periodic
from porewise.validation import decode_mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', nargs='?', help='segmented .tif, .tiff or .npy image')
    p.add_argument('--demo', action='store_true')
    p.add_argument('--backend', default='numpy')
    p.add_argument('--dx', type=float, required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--output', default='lbm-output-2d')
    p.add_argument('--solid-value', type=float)
    p.add_argument('--pore-value', type=float)
    a = p.parse_args(argv)
    if a.demo:
        blocked = geometry.parallel_plates(16, 24, 12)
    elif a.image:
        image = load_image(a.image)
        blocked = image if a.solid_value is None else decode_mask(
            image, solid_value=a.solid_value, pore_value=a.pore_value)
    else:
        return 0
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    permeabilities = {}
    for axis, force in zip('xy', ((1e-6, 0.), (0., 1e-6))):
        run = periodic(blocked, force, backend=a.backend, n_steps_max=a.steps,
                       quality_checks=False, return_fields=True, verbose=False)
        permeabilities[f'k_{axis}_m2'] = float(np.float32(run['k_lu'] * a.dx**2))
        np.savez_compressed(out / f'velocity_{axis}.npz', ux=np.asarray(run['ux'], dtype=np.float32),
                            uy=np.asarray(run['uy'], dtype=np.float32))
    (out / 'permeability.json').write_text(json.dumps(permeabilities, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
