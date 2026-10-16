import json

import numpy as np
import pytest

from porewise import save_permeability_excel
from porewise.__main__ import main

openpyxl = pytest.importorskip('openpyxl')


def read_rows(path):
    workbook = openpyxl.load_workbook(path)
    try:
        return list(workbook['Permeability'].values)
    finally:
        workbook.close()


def test_script_export_preserves_names_paths_and_numbers(tmp_path):
    result = dict(k_x_m2=np.float32(1e-12), k_y_m2=np.float32(2e-12), k_z_m2=0.)
    path = tmp_path / 'nested' / 'summary.xlsx'
    save_permeability_excel(path, [('first/rock.npy', result), ('second/rock.npy', result),
                                   ('=image.npy', result)])
    rows = read_rows(path)
    assert rows[0] == ('Image filename', 'Image path', 'kx (m²)', 'ky (m²)', 'kz (m²)')
    assert rows[1][:2] == ('rock.npy', 'first/rock.npy')
    assert rows[2][:2] == ('rock.npy', 'second/rock.npy')
    assert rows[3][0] == '=image.npy'
    assert rows[1][2:] == pytest.approx([float(result[f'k_{axis}_m2']) for axis in 'xyz'])
    save_permeability_excel(path, [('replacement.npy', result)])
    assert len(read_rows(path)) == 2


def test_cli_batch_excel_matches_per_image_json(tmp_path):
    images = [tmp_path / 'rock_one.npy', tmp_path / 'rock_two.npy']
    for index, image in enumerate(images):
        solid = np.ones((4, 6, 8), dtype=bool)
        solid[:, 1:3 + index] = False
        np.save(image, solid)
    out = tmp_path / 'results'
    summary = out / 'summary.xlsx'
    assert main([*map(str, images), '--backend', 'numpy', '--dx', '1e-6',
                 '--steps', '2', '--output', str(out), '--excel', str(summary)]) == 0
    rows = read_rows(summary)
    assert len(rows) == 3
    for index, image in enumerate(images, start=1):
        result = json.loads((out / f'{index:03d}_{image.stem}' / 'permeability.json').read_text())
        assert rows[index][:2] == (image.name, str(image))
        assert rows[index][2:] == pytest.approx([result[f'k_{axis}_m2'] for axis in 'xyz'])


def test_cli_excel_keeps_completed_rows_on_later_failure(tmp_path):
    image = tmp_path / 'first.npy'
    np.save(image, np.ones((3, 4, 5), dtype=bool))
    summary = tmp_path / 'summary.xlsx'
    with pytest.raises(FileNotFoundError):
        main([str(image), str(tmp_path / 'missing.npy'), '--backend', 'numpy',
              '--dx', '1e-6', '--steps', '1', '--output', str(tmp_path / 'out'),
              '--excel', str(summary)])
    assert read_rows(summary)[1] == (image.name, str(image), 0, 0, 0)
