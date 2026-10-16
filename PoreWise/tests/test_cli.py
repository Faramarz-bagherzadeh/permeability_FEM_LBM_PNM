import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from porewise.__main__ import main
from porewise.directional import compute_directional_permeability
from porewise import geometry
from porewise import solver


def test_cli_only_requested_outputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('numerical quality monitor must not run')
    monkeypatch.setattr(solver, 'Monitor', forbidden)
    blocked = np.ones((6, 8, 10), bool)
    blocked[:, 2:6, :] = False
    np.save(tmp_path / 'image.npy', blocked)
    out = tmp_path / 'results'
    assert main([str(tmp_path / 'image.npy'), '--backend', 'numpy', '--dx', '2e-6',
                 '--steps', '1', '--output', str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {
        'permeability.json', 'fields.vti'}
    permeabilities = json.loads((out / 'permeability.json').read_text())
    assert set(permeabilities) == {'k_x_m2', 'k_y_m2', 'k_z_m2'}
    arrays, image = read_fields_vti(out / 'fields.vti', blocked.shape)
    assert image.attrib['WholeExtent'] == '0 9 0 7 0 5'
    assert image.attrib['Spacing'] == '2e-06 2e-06 2e-06'
    assert set(arrays) == {'MicroCT', 'Velocity_X', 'Velocity_Y', 'Velocity_Z'}
    np.testing.assert_array_equal(arrays['MicroCT'], blocked.astype(np.float32))
    for axis in 'xyz':
        field = arrays['Velocity_' + axis.upper()]
        assert field.dtype == np.float32 and field.shape == blocked.shape + (3,)
        assert np.all(field[blocked] == 0)
        expected = ((1. - .5) / 3) * field[..., 'xyz'.index(axis)].mean(dtype=np.float64) / 1e-6 * (2e-6)**2
        assert permeabilities[f'k_{axis}_m2'] == pytest.approx(expected)


def read_fields_vti(path, shape):
    header, binary = path.read_bytes().split(b'<AppendedData encoding="raw">_', 1)
    root = ET.fromstring(header + b'</VTKFile>')
    arrays = {}
    for node in root.findall('.//PointData/DataArray'):
        assert node.attrib['type'] == 'Float32'
        offset = int(node.attrib['offset'])
        size = struct.unpack_from('<Q', binary, offset)[0]
        components = int(node.attrib['NumberOfComponents'])
        assert size == np.prod(shape) * components * 4
        values = np.frombuffer(binary, dtype='<f4', count=size // 4, offset=offset + 8)
        arrays[node.attrib['Name']] = values.reshape(shape + ((components,) if components > 1 else ()))
    return arrays, root.find('ImageData')


def test_three_independent_loads_and_physical_scaling():
    blocked = np.broadcast_to(geometry.parallel_plates(8, 10, 4), (6, 8, 10)).copy()
    settings = dict(backend='numpy', n_steps_max=600, verbose=False)
    result = compute_directional_permeability(blocked, 2e-6, **settings)
    scaled = compute_directional_permeability(blocked, 4e-6, **settings)
    assert set(result) == {'k_x_m2', 'k_y_m2', 'k_z_m2', 'velocity_fields'}
    # Discrete channel permeability uses total sample volume, including solid voxels.
    trt = compute_directional_permeability(blocked, 2e-6, collision='trt', **settings)
    assert trt['k_x_m2'] == pytest.approx((4**3 / 12 + 4 / 24) / 8 * (2e-6)**2, rel=1e-7)
    assert trt['k_z_m2'] == pytest.approx(trt['k_x_m2'], rel=1e-9)
    for axis in 'xyz':
        assert scaled[f'k_{axis}_m2'] == pytest.approx(4 * result[f'k_{axis}_m2'])
    assert isinstance(result['k_x_m2'], np.float32)
    assert all(component.dtype == np.float32 for field in result['velocity_fields'].values() for component in field.values())
    assert result['velocity_fields']['x']['ux'].max() > 0
    assert result['velocity_fields']['z']['uz'].max() > 0
    assert not np.allclose(result['velocity_fields']['x']['ux'], result['velocity_fields']['z']['ux'], atol=1e-10)


def test_batch_images_separate_outputs(tmp_path):
    images = [tmp_path / 'a.npy', tmp_path / 'b.npy']
    for image in images:
        np.save(image, np.ones((3, 4, 5), bool))
    out = tmp_path / 'batch'
    assert main([*map(str, images), '--backend', 'numpy', '--dx', '1e-6', '--steps', '1',
                 '--output', str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {'001_a', '002_b'}
    for directory in out.iterdir():
        assert all(value == 0 for value in json.loads((directory / 'permeability.json').read_text()).values())


def test_sparse_directional_fields_match_dense():
    pytest.importorskip('numba')
    blocked = np.broadcast_to(geometry.parallel_plates(8, 5, 4), (3, 8, 5)).copy()
    settings = dict(n_steps_max=600, verbose=False, collision='trt')
    dense = compute_directional_permeability(blocked, 1e-6, backend='numpy', **settings)
    sparse = compute_directional_permeability(blocked, 1e-6, backend='numba-sparse', **settings)
    for axis in 'xyz':
        assert sparse[f'k_{axis}_m2'] == pytest.approx(dense[f'k_{axis}_m2'], rel=1e-7, abs=1e-22)
        for component in ('ux', 'uy', 'uz'):
            np.testing.assert_allclose(sparse['velocity_fields'][axis][component],
                                       dense['velocity_fields'][axis][component], atol=1e-14)


@pytest.mark.parametrize('mirror', [False, True])
def test_vti_preserves_npy_labels_and_mirrored_alignment(tmp_path, mirror):
    image = np.full((3, 4, 5), 255, dtype=np.uint8)
    image[:, 1:3] = 0
    np.save(tmp_path / 'image.npy', image)
    out = tmp_path / 'output'
    args = [str(tmp_path / 'image.npy'), '--backend', 'numpy', '--dx', '1e-6',
            '--solid-value', '255', '--pore-value', '0', '--steps', '2', '--output', str(out)]
    if mirror:
        args += ['--mirror']
    assert main(args) == 0
    expected = image
    if mirror:
        for axis in range(3):
            expected = np.concatenate([expected, np.flip(expected, axis=axis)], axis=axis)
    arrays, _ = read_fields_vti(out / 'fields.vti', expected.shape)
    np.testing.assert_array_equal(arrays['MicroCT'], expected.astype(np.float32))
    for axis in 'XYZ':
        assert np.all(arrays['Velocity_' + axis][expected == 255] == 0)
