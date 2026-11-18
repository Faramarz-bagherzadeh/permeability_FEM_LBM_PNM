# permeability_FEM_LBM_PNM

The code repository for calculating and comparing permeability with different methods.

The CHFEM workflow fills void components that do not connect to any image
boundary before solving permeability. VTI exports retain the original binary
image. The helper is also available as `isolated_voids.fill_isolated_voids`.

