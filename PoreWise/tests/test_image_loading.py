"""TIFF inputs preserve labels and avoid intermediate disk copies."""
import weakref

import numpy as np
import pytest
import tifffile

from porewise import load_image
from porewise.__main__ import main
import porewise.__main__ as cli
from tests.test_cli import read_fields_vti


@pytest.mark.parametrize('suffix', ['.tif', '.tiff', '.TIF', '.TIFF', '.npy'])
def test_image_loading_preserves_dtype_shape_and_labels(tmp_path, suffix):
    image = np.array([0, 1, 255, 4096, 65535, 128], dtype=np.uint16).reshape(1, 2, 3)
    path = tmp_path / ('image' + suffix)
    if suffix == '.npy':
        np.save(path, image)
    else:
        tifffile.imwrite(path, image, photometric='minisblack')
    result = load_image(path)
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, image)
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_image_format(tmp_path):
    with pytest.raises(ValueError, match='Unsupported image format'):
        load_image(tmp_path / 'image.png')


@pytest.mark.parametrize('mirror', [False, True])
def test_tiff_and_npy_produce_identical_directional_exports(tmp_path, mirror):
    image = np.full((3, 4, 5), 4096, dtype=np.uint16)
    image[:, 1:3] = 0
    tifffile.imwrite(tmp_path / 'image.tif', image, photometric='minisblack')
    np.save(tmp_path / 'image.npy', image)
    for suffix in ('tif', 'npy'):
        args = [str(tmp_path / f'image.{suffix}'), '--backend', 'numpy',
                '--dx', '1e-6', '--solid-value', '4096', '--pore-value', '0',
                '--steps', '2', '--output', str(tmp_path / suffix)]
        if mirror:
            args += ['--mirror']
        assert main(args) == 0
    for name in ('permeability.json', 'fields.vti'):
        assert (tmp_path / 'tif' / name).read_bytes() == (tmp_path / 'npy' / name).read_bytes()
    expected = image
    if mirror:
        for axis in range(3):
            expected = np.concatenate([expected, np.flip(expected, axis=axis)], axis=axis)
    arrays, _ = read_fields_vti(tmp_path / 'tif/fields.vti', expected.shape)
    np.testing.assert_array_equal(arrays['MicroCT'], expected.astype(np.float32))


def test_mixed_batch_exports_before_loading_next_image(tmp_path, monkeypatch):
    paths = [tmp_path / 'a.tif', tmp_path / 'b.tiff', tmp_path / 'c.npy']
    image = np.ones((3, 4, 5), dtype=np.uint8)
    for path in paths:
        if path.suffix == '.npy':
            np.save(path, image)
        else:
            tifffile.imwrite(path, image, photometric='minisblack')
    output = tmp_path / 'results'
    references = []

    def load(path):
        if references:
            assert references[-1]() is None, 'previous image retained during next load'
            previous = output / f'{len(references):03d}_{paths[len(references) - 1].stem}'
            assert (previous / 'fields.vti').is_file()
            assert (previous / 'permeability.json').is_file()
        result = load_image(path)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(cli, 'load_image', load)
    assert main([*map(str, paths), '--backend', 'numpy', '--dx', '1e-6',
                 '--solid-value', '1', '--pore-value', '0', '--steps', '1',
                 '--output', str(output)]) == 0
    assert all(reference() is None for reference in references)
    assert list(tmp_path.glob('*.npy')) == [paths[-1]]
    assert {path.name for path in output.iterdir()} == {'001_a', '002_b', '003_c'}
