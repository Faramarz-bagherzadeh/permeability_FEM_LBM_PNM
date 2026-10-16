"""Compute x/y/z permeability (m²) and a Float32 VTI with image and three velocity fields."""
import argparse
from pathlib import Path

import numpy as np

from . import geometry
from .directional import save_directional_results
from .excel import save_permeability_excel
from .io import load_image
from .validation import decode_mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='*', help='segmented 3D .tif, .tiff or .npy images; loaded one at a time')
    p.add_argument('--demo', action='store_true')
    p.add_argument('--backend', default='auto')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--precision', default='float64')
    p.add_argument('--F', type=float, default=1e-6)
    p.add_argument('--tau', type=float, default=1.)
    p.add_argument('--collision', default='bgk')
    p.add_argument('--magic', type=float)
    p.add_argument('--steps', type=int, default=20000, help='fixed updates per driving direction')
    p.add_argument('--mirror', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--heartbeat', type=int, default=1000)
    p.add_argument('--dx', type=float, required=True, help='voxel size in metres')
    p.add_argument('--output', default='lbm-output')
    p.add_argument('--excel', type=Path, metavar='SUMMARY.xlsx',
                   help='also save one summary row per image with x/y/z permeability in m²')
    p.add_argument('--solid-value', type=float)
    p.add_argument('--pore-value', type=float)
    a = p.parse_args(argv)
    settings = dict(backend='numpy' if a.no_gpu else a.backend, precision=a.precision,
                    tau=a.tau, collision=a.collision, magic=a.magic,
                    n_steps_max=a.steps, verbose=a.verbose, heartbeat=a.heartbeat,
                    force_magnitude=a.F, mirror=a.mirror)
    images = a.images or ([None] if a.demo else [])
    excel_results = []
    for index, filename in enumerate(images):
        if filename is None:
            blocked = np.broadcast_to(geometry.parallel_plates(16, 24, 12), (8, 16, 24)).copy()
            image = blocked
        else:
            image = load_image(filename)
            blocked = image if a.solid_value is None else decode_mask(
                image, solid_value=a.solid_value, pore_value=a.pore_value)
        directory = Path(a.output)
        if len(images) > 1:
            directory = directory / f'{index + 1:03d}_{Path(filename).stem}'
        result = save_directional_results(directory, blocked, a.dx, microct=image, **settings)
        if a.excel is not None:
            excel_results.append((filename if filename is not None else 'demo', result))
            save_permeability_excel(a.excel, excel_results)
        print(f'Permeabilities and image with three Float32 velocity fields saved in {directory}')
        del image, blocked, result
    if a.excel is not None:
        print(f'Excel permeability summary saved in {a.excel}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
