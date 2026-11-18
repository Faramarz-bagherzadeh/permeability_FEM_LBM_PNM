import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import chfem_permeability
from isolated_voids import fill_isolated_voids


class FillIsolatedVoidsTests(unittest.TestCase):
    def test_fills_enclosed_void_but_preserves_boundary_connected_void(self):
        image = np.ones((5, 5, 5), dtype=np.uint8)
        image[2, 2, 2] = 0
        image[0, 0, 0] = 0
        image[1, 0, 0] = 0
        original = image.copy()

        filled = fill_isolated_voids(image)

        self.assertEqual(filled[2, 2, 2], 1)
        self.assertEqual(filled[0, 0, 0], 0)
        self.assertEqual(filled[1, 0, 0], 0)
        np.testing.assert_array_equal(image, original)

    def test_requires_a_three_dimensional_binary_image(self):
        with self.assertRaises(ValueError):
            fill_isolated_voids(np.zeros((3, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            fill_isolated_voids(np.full((3, 3, 3), 2, dtype=np.uint8))

    def test_solver_receives_filled_image_but_vti_receives_original(self):
        image = np.ones((5, 5, 5), dtype=np.uint8)
        image[2, 2, 2] = 0
        image[0, 0, 0] = 0
        image[1, 0, 0] = 0
        solver_inputs = []
        vti_images = []
        fake_chfem = types.SimpleNamespace(
            compute_property=lambda _name, domain, **_kwargs: (
                solver_inputs.append(domain.copy()) or 1.0
            ),
            import_vector_field_from_chfem=lambda _path, shape: np.zeros(
                (3, *shape), dtype=np.float32
            ),
        )

        def capture_vti(_path, vti_image, _velocity_fields, _voxel_size):
            vti_images.append(vti_image.copy())

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(sys.modules, {"chfem": fake_chfem}), patch.object(
                chfem_permeability, "_write_vti", new=capture_vti
            ):
                chfem_permeability.compute_directional_permeability(
                    image, velocity_output_dir=output_dir
                )

        self.assertEqual(len(solver_inputs), 3)
        for solver_input in solver_inputs:
            self.assertEqual(solver_input[2, 2, 2], 1)
            self.assertEqual(solver_input[0, 0, 0], 0)
        self.assertEqual(len(vti_images), 1)
        np.testing.assert_array_equal(vti_images[0], image)


if __name__ == "__main__":
    unittest.main()
