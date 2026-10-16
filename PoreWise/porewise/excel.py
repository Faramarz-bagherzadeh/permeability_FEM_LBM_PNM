"""Excel summaries of directional permeability, with one row per source image."""
from pathlib import Path


def _new_workbook():
    from openpyxl import Workbook
    return Workbook()


def save_permeability_excel(path, results):
    """Save (image_path, result) pairs to an .xlsx workbook, replacing that file.

    Each result supplies k_x_m2, k_y_m2 and k_z_m2, as returned by
    compute_directional_permeability or save_directional_results. Only scalar
    permeabilities are exported; velocity fields are ignored. Paths distinguish
    images with the same filename. Existing parent directories are created.
    """
    path = Path(path)
    workbook = _new_workbook()
    sheet = workbook.active
    sheet.title = 'Permeability'
    sheet.append(['Image filename', 'Image path', 'kx (m²)', 'ky (m²)', 'kz (m²)'])
    for image_path, result in results:
        image_path = Path(image_path)
        # Explicit strings keep filenames beginning with '=' from becoming formulas.
        row = sheet.max_row + 1
        for column, value in enumerate((image_path.name, str(image_path)), start=1):
            cell = sheet.cell(row, column, value)
            cell.data_type = 's'
        for column, axis in enumerate('xyz', start=3):
            cell = sheet.cell(row, column, float(result[f'k_{axis}_m2']))
            cell.number_format = '0.000000E+00'
    sheet.freeze_panes = 'C2'
    sheet.auto_filter.ref = sheet.dimensions
    for column, width in [('A', 48), ('B', 80), ('C', 22), ('D', 22), ('E', 22)]:
        sheet.column_dimensions[column].width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.xlsx')
    try:
        workbook.save(temporary)
        temporary.replace(path)
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)
