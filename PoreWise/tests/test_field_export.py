import numpy as np
import pytest
from porewise.field_export import write_vti


def test_vti_noncubic_roundtrip(tmp_path):
    vtk = pytest.importorskip('vtk')
    from vtk.util.numpy_support import vtk_to_numpy
    z,y,x = np.indices((3,4,5)); mask=(x==2)&(y==1)
    fields=dict(ux=(100*z+10*y+x).astype(float),uy=np.full(mask.shape,-2.),
                uz=np.full(mask.shape,7.),rho=np.ones(mask.shape))
    path=tmp_path/'field.vti'
    metadata=write_vti(path,mask,fields,2e-6,origin=(1e-3,2e-3,3e-3))
    reader=vtk.vtkXMLImageDataReader();reader.SetFileName(str(path));reader.Update()
    image=reader.GetOutput()
    assert image.GetDimensions()==(6,5,4)
    assert image.GetNumberOfCells()==60 and image.GetPointData().GetNumberOfArrays()==0
    np.testing.assert_allclose(image.GetSpacing(),[2e-6]*3)
    np.testing.assert_allclose(image.GetOrigin(),[1e-3,2e-3,3e-3])
    vel=vtk_to_numpy(image.GetCellData().GetArray('velocity_lu')).reshape(3,4,5,3)
    for i,c in enumerate('xyz'):np.testing.assert_array_equal(vel[...,i],fields['u'+c])
    np.testing.assert_array_equal(vtk_to_numpy(image.GetCellData().GetArray('solid')).reshape(mask.shape),mask)
    np.testing.assert_allclose(image.GetCell(0).GetBounds(),[.001,.001002,.002,.002002,.003,.003002])
    assert metadata['association']=='cell'




def test_directional_vti_roundtrip_with_vtk(tmp_path):
    vtk = pytest.importorskip('vtk')
    from vtk.util.numpy_support import vtk_to_numpy
    from porewise.field_export import write_directional_vti
    z, y, x = np.indices((3, 4, 5))
    microct = np.where(x == 2, 255, 0).astype(np.uint8)
    fields = [dict(ux=100*z + 10*y + x + j, uy=np.full(microct.shape, -2-j),
                   uz=np.full(microct.shape, 7+j)) for j in range(3)]
    path = tmp_path / 'fields.vti'
    k = write_directional_vti(path, microct, ((axis, (j+1)*1e-12, fields[j])
                              for j, axis in enumerate('xyz')), 2e-6)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == (5, 4, 3)
    assert image.GetNumberOfPoints() == 60
    assert image.GetCellData().GetNumberOfArrays() == 0
    np.testing.assert_allclose(image.GetSpacing(), [2e-6]*3)
    assert image.GetOrigin() == (0., 0., 0.)
    data = image.GetPointData()
    assert data.GetNumberOfArrays() == 4
    assert data.GetScalars().GetName() == 'MicroCT'
    assert data.GetVectors().GetName() == 'Velocity_X'
    scalar = vtk_to_numpy(data.GetArray('MicroCT'))
    assert scalar.dtype == np.float32
    np.testing.assert_array_equal(scalar.reshape(microct.shape), microct)
    for j, axis in enumerate('XYZ'):
        vector = vtk_to_numpy(data.GetArray('Velocity_' + axis)).reshape(microct.shape + (3,))
        assert vector.dtype == np.float32
        for i, component in enumerate('xyz'):
            np.testing.assert_array_equal(vector[..., i], fields[j]['u' + component])
        assert k[f'k_{axis.lower()}_m2'] == float(np.float32((j+1)*1e-12))
