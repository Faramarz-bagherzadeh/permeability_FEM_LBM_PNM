#!/usr/bin/env python3
"""Pressure-driven D3Q19 BGK permeability of binary (z,y,x) TIFF volumes.

The OpenLB example uses pore-only SuperAverage3D, including its porosity factor
in the reported K. This module preserves that definition for comparison.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label

DIRECTIONS = ("x", "y", "z")
# Solver order is (z,y,x), velocity order is (ux,uy,uz).
AXES = {"x": (0, 1, 2), "y": (0, 2, 1), "z": (1, 2, 0)}
C = np.array([(0,0,0), (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),
              (0,0,1),(0,0,-1),(1,1,0),(-1,-1,0),(1,-1,0),(-1,1,0),
              (1,0,1),(-1,0,-1),(1,0,-1),(-1,0,1),
              (0,1,1),(0,-1,-1),(0,1,-1),(0,-1,1)], dtype=np.int8)
W = np.array([1/3] + [1/18]*6 + [1/36]*12)
OPP = np.array([next(j for j, b in enumerate(C) if np.array_equal(b, -a)) for a in C])
CX = C[:, 0]


def load_segmented_tiff(path: str | Path) -> np.ndarray:
    volume = tifffile.imread(path)
    if volume.ndim != 3 or not np.isin(volume, (0, 1)).all():
        raise ValueError(f"{path}: expected 3-D binary TIFF with 0=pore, 1=solid")
    if not np.any(volume == 0):
        raise ValueError(f"{path}: no pore voxels")
    return volume


def orient_geometry(volume: np.ndarray, direction: str) -> np.ndarray:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction}")
    return np.transpose(volume, AXES[direction])


def _equilibrium(rho: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    cu = np.einsum('ia,...a->i...', C, velocity, optimize=True)
    u2 = np.sum(velocity**2, axis=-1)
    return W.reshape((-1,) + (1,)*rho.ndim) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)


def _connected(pore: np.ndarray) -> bool:
    # Face-connected pore paths, not corner-only contacts.
    components, _ = label(pore, structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                                     [[0,1,0],[1,1,1],[0,1,0]],
                                                     [[0,0,0],[0,1,0],[0,0,0]]]))
    return bool(np.any(np.intersect1d(np.unique(components[..., 0]),
                               np.unique(components[..., -1]), assume_unique=True) > 0))


def run_lbm(volume_zyx: np.ndarray, *, voxel_size_m: float = 120e-6,
            density_kg_m3: float = 1000., kinematic_viscosity_m2_s: float = 1e-6,
            pressure_drop_pa: float = .01, tau: float = .75,
            max_iterations: int = 20000, tolerance: float = 1e-6,
            check_interval: int = 100, max_voxels: int = 65_000_000) -> dict:
    """Solve along x with pressure planes and periodic transverse faces.

    Pressure planes are one lattice site outside the unmodified sample. Incoming
    populations use a pressure/unknown-normal-velocity bounce reconstruction.
    This approximates OpenLB's regularized LocalPressure boundary; the interior
    D3Q19 BGK and halfway solid bounce-back match its model.
    """
    for name, value in (("voxel_size_m",voxel_size_m),("density_kg_m3",density_kg_m3),
                        ("kinematic_viscosity_m2_s",kinematic_viscosity_m2_s),
                        ("pressure_drop_pa",pressure_drop_pa), ("tolerance",tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not .5 < tau < 2 or max_iterations < 1 or check_interval < 1:
        raise ValueError("require 0.5 < tau < 2 and positive iteration settings")
    if volume_zyx.ndim != 3 or not np.isin(volume_zyx,(0,1)).all():
        raise ValueError("expected 3-D binary volume")
    if volume_zyx.size > max_voxels:
        raise MemoryError(f"{volume_zyx.size:,} voxels exceed max_voxels={max_voxels:,}; "
                          "D3Q19 NumPy needs several GB per tens of millions of voxels")
    pore = volume_zyx == 0
    if not _connected(pore):
        raise ValueError("no face-connected inlet-to-outlet pore path")
    nz,ny,nx = pore.shape
    if nx < 3:
        raise ValueError("flow direction must contain at least three voxels")
    dx = voxel_size_m
    nu_lattice = (tau-.5)/3
    dt = nu_lattice*dx*dx/kinematic_viscosity_m2_s
    velocity_scale = dx/dt
    # p_phys = rho_phys*c_s^2*(dx/dt)^2*(rho_lattice-1).
    delta_rho = pressure_drop_pa/(density_kg_m3*(velocity_scale**2)/3)
    if delta_rho <= 0 or delta_rho > .02:
        raise ValueError(f"pressure drop gives lattice density difference {delta_rho:.3g}; "
                         "reduce pressure_drop_pa or adjust tau for low Mach flow")
    solid = np.ones((nz,ny,nx+2), dtype=bool)
    solid[...,1:-1] = ~pore
    solid[...,0] = ~pore[...,0]
    solid[...,-1] = ~pore[...,-1]
    fluid = ~solid
    rho = np.ones(solid.shape)
    u = np.zeros(solid.shape+(3,))
    f = _equilibrium(rho,u)
    previous_energy = None
    residual = math.inf
    converged = False
    for iteration in range(1,max_iterations+1):
        rho = f.sum(axis=0)
        momentum = np.einsum('i...,ia->...a',f,C,optimize=True)
        u = np.divide(momentum,rho[...,None],out=np.zeros_like(momentum),where=rho[...,None]>0)
        u[solid] = 0
        if not np.isfinite(u).all() or np.max(np.abs(u[fluid])) > .15:
            raise FloatingPointError("LBM diverged or exceeded low-Mach limit")
        equilibrium = _equilibrium(rho,u)
        f_post = f - (f-equilibrium)/tau
        streamed = np.empty_like(f)
        for i,(cx,cy,cz) in enumerate(C):
            shifted = np.roll(f_post[i], (int(cz),int(cy)), axis=(0,1))
            if cx == 1:
                streamed[i,...,1:] = shifted[...,:-1]
                streamed[i,...,0] = f_post[OPP[i],...,0]
            elif cx == -1:
                streamed[i,...,:-1] = shifted[...,1:]
                streamed[i,...,-1] = f_post[OPP[i],...,-1]
            else:
                streamed[i] = shifted
            upstream_solid = np.roll(solid,(int(cz),int(cy),int(cx)),axis=(0,1,2))
            bounced = fluid & upstream_solid
            streamed[i,bounced] = f_post[OPP[i],bounced]
        f = streamed
        # Zou/He pressure reconstruction on open reservoir faces. Pressure is
        # ramped during the first fifth of the run, as in the OpenLB example.
        rho_in = 1 + delta_rho * min(1.,iteration/max(1,max_iterations//5))
        for side, target_rho, incoming, outgoing in ((0,rho_in,1,-1),(-1,1.,-1,1)):
            active = fluid[...,side]
            plane = f[...,side]
            zero = plane[CX==0].sum(axis=0)
            known = plane[CX==outgoing].sum(axis=0)
            ux = (1-(zero+2*known)/target_rho) if side == 0 else (-1+(zero+2*known)/target_rho)
            for i in np.flatnonzero(CX==incoming):
                # Tangential non-equilibrium is copied from the opposite link.
                reconstructed = plane[OPP[i]] + 6*W[i]*target_rho*CX[i]*ux
                plane[i,active] = reconstructed[active]
        if iteration % check_interval == 0 and iteration > max_iterations//5:
            energy = float(np.mean(np.sum(u[fluid]**2,axis=-1)))
            if previous_energy is not None:
                residual = abs(energy-previous_energy)/max(energy,1e-30)
                if residual < tolerance:
                    converged = True
                    break
            previous_energy = energy
    rho = f.sum(axis=0)
    u = np.divide(np.einsum('i...,ia->...a',f,C,optimize=True),rho[...,None],
                  out=np.zeros(solid.shape+(3,)),where=rho[...,None]>0)
    physical_velocity = u[...,1:-1,:]*velocity_scale
    physical_velocity[~pore] = 0
    return {"velocity":physical_velocity,"converged":converged,
            "iterations":iteration,"final_residual":residual,
            "dx_m":dx,"dt_s":dt,"nu_lattice":nu_lattice}


def calculate_permeability_for_image(image_path: str | Path, output_dir: str | Path,
                                     **solver_options) -> dict:
    volume = load_segmented_tiff(image_path)
    row = {"file":Path(image_path).name,"porosity":float(np.mean(volume==0))}
    out = Path(output_dir)/Path(image_path).stem
    out.mkdir(parents=True,exist_ok=True)
    for direction in DIRECTIONS:
        oriented = orient_geometry(volume,direction)
        prefix = direction
        try:
            result = run_lbm(oriented,**solver_options)
            axes = AXES[direction]
            inverse = np.argsort(axes)
            velocity = np.transpose(result["velocity"],tuple(inverse)+(3,))
            # Solver xyz components map to physical xyz via the oriented axes.
            physical_components = (2-axes[2],2-axes[1],2-axes[0])
            velocity = velocity[...,np.argsort(physical_components)]
            np.save(out/f"velocity_{direction}.npy",velocity)
            row[f"{prefix}_converged"] = result["converged"]
            row[f"{prefix}_iterations"] = result["iterations"]
            row[f"{prefix}_final_residual"] = result["final_residual"]
            if not result["converged"]:
                raise RuntimeError("LBM did not converge")
            # Match OpenLB SuperAverage3D(material 1): pore-only mean.
            mean_pore_velocity = float(np.mean(result["velocity"][...,0][oriented==0]))
            length_m = (oriented.shape[-1]+.5)*solver_options.get("voxel_size_m",120e-6)
            mu = solver_options.get("density_kg_m3",1000.)*solver_options.get("kinematic_viscosity_m2_s",1e-6)
            pressure_drop = solver_options.get("pressure_drop_pa",.01)
            k = mean_pore_velocity*mu*length_m/pressure_drop
            if not np.isfinite(k) or k <= 0:
                raise ValueError(f"non-positive or non-finite permeability: {k}")
            row[f"permeability_{direction}_m2"] = k
        except (ValueError,RuntimeError,FloatingPointError,MemoryError) as exc:
            row[f"permeability_{direction}_m2"] = math.nan
            row.setdefault(f"{prefix}_converged",False)
            row.setdefault(f"{prefix}_iterations",0)
            row.setdefault(f"{prefix}_final_residual",math.nan)
            row[f"{prefix}_error"] = str(exc)
    return row


def calculate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for d in DIRECTIONS:
        true = results[f"ground_truth_{d}_m2"].to_numpy(dtype=float)
        pred = results[f"permeability_{d}_m2"].to_numpy(dtype=float)
        valid = np.isfinite(true)&np.isfinite(pred)&(true>0)&(pred>0)
        a,b = np.log10(true[valid]),np.log10(pred[valid])
        error = b-a
        ss = np.sum((a-a.mean())**2) if len(a) else 0
        rows.append({"direction":d,"n":len(a),"R2":1-np.sum(error**2)/ss if len(a)>1 and ss>0 else math.nan,
                     "MAE_log10":float(np.mean(abs(error))) if len(a) else math.nan,
                     "RMSE_log10":float(np.sqrt(np.mean(error**2))) if len(a) else math.nan})
    return pd.DataFrame(rows)


def calculate_permeability_folder(image_folder: str | Path, ground_truth_path: str | Path,
                                  output_dir: str | Path = "results", **solver_options) -> tuple[pd.DataFrame,pd.DataFrame]:
    images = sorted(p for p in Path(image_folder).iterdir() if p.suffix.lower() in ('.tif','.tiff'))
    if not images:
        raise ValueError(f"no TIFF files in {image_folder}")
    truth = pd.read_excel(ground_truth_path)
    if 'filename' not in truth or truth['filename'].isna().any() or truth['filename'].duplicated().any():
        raise ValueError("ground truth needs unique nonempty filename values")
    needed = [f"permeability_{d}_m2" for d in DIRECTIONS]
    if any(c not in truth for c in needed):
        raise ValueError(f"ground truth needs {needed}")
    truth = truth.set_index('filename')
    missing = [p.name for p in images if p.name not in truth.index]
    if missing:
        raise ValueError(f"TIFF filenames absent from ground truth: {missing[:5]}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for image in images:
        row=calculate_permeability_for_image(image,output_dir/'velocity_fields',**solver_options)
        for d in DIRECTIONS:
            pred=row[f"permeability_{d}_m2"]
            gt=float(truth.loc[image.name,f"permeability_{d}_m2"])
            row[f"ground_truth_{d}_m2"]=gt
            row[f"log10_permeability_{d}"]=math.log10(pred) if np.isfinite(pred) and pred>0 else math.nan
            row[f"log10_ground_truth_{d}"]=math.log10(gt) if np.isfinite(gt) and gt>0 else math.nan
        rows.append(row)
    results=pd.DataFrame(rows)
    metrics=calculate_metrics(results)
    results.to_excel(output_dir/'permeability_results.xlsx',index=False)
    metrics.to_excel(output_dir/'permeability_metrics.xlsx',index=False)
    return results,metrics


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    parser.add_argument('--voxel-size',type=float,default=120e-6)
    parser.add_argument('--pressure-drop',type=float,default=.01)
    parser.add_argument('--max-iterations',type=int,default=20000)
    parser.add_argument('--max-voxels',type=int,default=65_000_000)
    args=parser.parse_args()
    results,metrics=calculate_permeability_folder(args.data_dir,args.ground_truth,args.output_dir,
        voxel_size_m=args.voxel_size,pressure_drop_pa=args.pressure_drop,
        max_iterations=args.max_iterations,max_voxels=args.max_voxels)
    print(results[['file']+[f'permeability_{d}_m2' for d in DIRECTIONS]].to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
