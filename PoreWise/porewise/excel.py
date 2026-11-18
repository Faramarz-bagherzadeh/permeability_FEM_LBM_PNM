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
    sheet.append([
        'Image filename', 'Image path',
        'kx (m²)', 'ky (m²)', 'kz (m²)',
        'Status x', 'Status y', 'Status z', 'Overall status',
        'Iterations x', 'Iterations y', 'Iterations z',
    ])
    for image_path, result in results:
        image_path = Path(image_path)
        # Explicit strings keep filenames beginning with '=' from becoming formulas.
        row = sheet.max_row + 1
        for column, value in enumerate((image_path.name, str(image_path)), start=1):
            cell = sheet.cell(row, column, value)
            cell.data_type = 's'
        for column, axis in enumerate('xyz', start=3):
            value = result[f'k_{axis}_m2']
            cell = sheet.cell(row, column, None if value is None else float(value))
            cell.number_format = '0.000000E+00'
        statuses = []
        for column, axis in enumerate('xyz', start=6):
            converged = result.get(f'converged_{axis}')
            if converged is True:
                status = 'Converged'
            elif converged is False:
                status = str(result.get(
                    f'termination_{axis}', 'Not converged'
                ))
            else:
                status = 'Not evaluated'
            statuses.append(converged)
            sheet.cell(row, column, status)
        if all(status is True for status in statuses):
            overall_status = 'Converged'
        elif any(status is False for status in statuses):
            overall_status = 'Not converged'
        else:
            overall_status = 'Not evaluated'
        sheet.cell(row, 9, overall_status)
        for column, axis in enumerate('xyz', start=10):
            sheet.cell(row, column, int(result.get(f'iterations_{axis}', 0)))
    sheet.freeze_panes = 'C2'
    sheet.auto_filter.ref = sheet.dimensions
    for column, width in [
        ('A', 48), ('B', 80), ('C', 22), ('D', 22), ('E', 22),
        ('F', 18), ('G', 18), ('H', 18), ('I', 18),
        ('J', 14), ('K', 14), ('L', 14),
    ]:
        sheet.column_dimensions[column].width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.xlsx')
    try:
        workbook.save(temporary)
        temporary.replace(path)
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)
