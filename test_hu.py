import json
import math
import pathlib
from dataclasses import dataclass

import gecatsim as xc
import numpy as np
import tomosipo as ts
import torch
from gecatsim.pyfiles.GetMu import GetMu
from tifffile import tifffile
from ts_algorithms import fdk

from make_projections import do_projs, get_config, make_projs
from make_slice_gt import cfg_to_str, create_config


@dataclass
class Cylinder:
    diameter: float
    height: float
    mat_name: str
    mat_value: float


@dataclass
class InnerCylinder:
    cylinder: Cylinder
    distance: float
    angle: float


@dataclass
class PhantomDesc:
    outer: Cylinder
    inner: list[InnerCylinder]


@dataclass
class Region:
    distance: float
    angle: float
    diameter: float
    slice: int


def make_vol_by_desc(desc: PhantomDesc, voxel_size: float) -> tuple[np.ndarray, list[np.ndarray]]:
    d_vox = int(desc.outer.diameter / voxel_size + 0.5)
    h_vox = int(desc.outer.height / voxel_size + 0.5)

    i_row, i_col = np.meshgrid(np.arange(d_vox), np.arange(d_vox), indexing="ij")
    center = (d_vox - 1) / 2

    r_outer = desc.outer.diameter / 2
    one_outer_slice = desc.outer.mat_value * np.float32(
        (voxel_size * (i_row - center)) ** 2 + (voxel_size * (i_col - center)) ** 2 <= r_outer * r_outer,
    )

    inner_vols = []
    for inner in desc.inner:
        dist = inner.distance
        angle = inner.angle
        r_inner = inner.cylinder.diameter / 2
        x_c = r_outer + dist * math.cos(angle)
        y_c = r_outer - dist * math.sin(angle)
        one_inner_slice = inner.cylinder.mat_value * np.float32(
            (voxel_size * i_col - x_c) ** 2 + (voxel_size * i_row - y_c) ** 2 <= r_inner * r_inner,
        )
        one_outer_slice[one_inner_slice > 0] = 0.0
        inner_vols.append(np.repeat(one_inner_slice[np.newaxis, :, :], h_vox, axis=0))

    return np.repeat(one_outer_slice[np.newaxis, :, :], h_vox, axis=0), inner_vols


def make_phantom_by_desc(desc: PhantomDesc, voxel_size: float, phantom_path: pathlib.Path) -> tuple[int, int]:
    phantom_path.mkdir(exist_ok=True, parents=True)
    outer_vol, inner_vols = make_vol_by_desc(desc, voxel_size)

    vol_materials = {desc.outer.mat_name: outer_vol}
    for inner, inner_vol in zip(desc.inner, inner_vols):
        if inner.cylinder.mat_name in vol_materials:
            vol_materials[inner.cylinder.mat_name] = np.where(
                inner_vol > 0, inner_vol, vol_materials[inner.cylinder.mat_name]
            )
        else:
            vol_materials[inner.cylinder.mat_name] = inner_vol

    h_vox = outer_vol.shape[0]
    d_vox = outer_vol.shape[1]

    vol_file_names = []
    for mat_name, vol in vol_materials.items():
        basename = f"phantom_{mat_name}"
        vol_file_names.append(f"{basename}.raw")
        tifffile.imwrite(phantom_path / f"{basename}.tif", vol, imagej=True, compression="zlib")
        xc.rawwrite(str(phantom_path / f"{basename}.raw"), vol)

    json_cfg = create_config(
        [h_vox, d_vox, d_vox],
        (voxel_size, voxel_size, voxel_size),
        vol_file_names,
        list(vol_materials.keys()),
        50,
    )
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(json_cfg, f, indent=4)

    return d_vox, h_vox


def make_projections(cfg_path: pathlib.Path, phantom_path: pathlib.Path) -> xc.CatSim:
    phantom_json_path = phantom_path / "phantom.json"
    xcist = do_projs(
        phantom_json_path,
        get_config(cfg_path),
        phantom_path,
    )

    num = xcist.cfg.protocol.viewCount
    rows = xcist.cfg.scanner.detectorRowCount
    cols = xcist.cfg.scanner.detectorColCount
    projs = xc.rawread(str(phantom_path / "projs.prep"), (num, rows, cols), "float")
    tifffile.imwrite(phantom_path / f"projs.tif", projs, imagej=True, compression="zlib")

    return xcist


def make_reconstruct(projs_path: pathlib.Path, xcist: xc.CatSim, voxel_size: float) -> np.ndarray:
    num = int(xcist.cfg.protocol.viewCount)
    rows = int(xcist.cfg.scanner.detectorRowCount)
    cols = int(xcist.cfg.scanner.detectorColCount)
    dso = xcist.cfg.scanner.sid
    dsd = xcist.cfg.scanner.sdd
    image_shape = [rows, cols, cols]
    image_size = list(map(lambda x: x * voxel_size, image_shape))

    projs = tifffile.imread(projs_path)
    assert num == projs.shape[0]

    det_pixel_size = xcist.cfg.scanner.detectorColSize

    detector_shape = [projs.shape[1], projs.shape[2]]
    detector_size = [detector_shape[0] * det_pixel_size, detector_shape[1] * det_pixel_size]
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)

    vg = ts.volume(shape=np.array(image_shape), size=np.array(image_size, dtype=np.float64))
    pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
    A = ts.operator(vg, pg)

    sino = np.transpose(projs, (1, 0, 2))
    sino = torch.from_numpy(sino)  # .cuda()

    recon = fdk(A, sino)
    recon = recon.detach().cpu().numpy()

    return recon


def test(
    volume: np.ndarray, voxel_size: float, xcist: xc.CatSim, lcr_list: list[Region], lc_list: list[Region]
) -> None:
    mu_water = GetMu("water", xcist.cfg.physics.monochromatic)
    vol_hu = 1000 * (10 * volume - mu_water[0]) / mu_water[0]
    shape = vol_hu.shape

    lcr_hu = []
    for lcr in lcr_list:
        x, y = (
            shape[1] * voxel_size / 2 + lcr.distance * math.cos(lcr.angle),
            shape[2] * voxel_size / 2 - lcr.distance * math.sin(lcr.angle),
        )
        x_i, y_i = int(x / voxel_size), int(y / voxel_size)
        r = int(lcr.diameter / 2 / voxel_size)
        lcr_hu.append(np.mean(vol_hu[lcr.slice, y_i - r : y_i + r, x_i - r : x_i + r]))

    lc_hu = []
    for lc in lc_list:
        x, y = (
            shape[1] * voxel_size / 2 + lc.distance * math.cos(lc.angle),
            shape[2] * voxel_size / 2 - lc.distance * math.sin(lc.angle),
        )
        x_i, y_i = int(x / voxel_size), int(y / voxel_size)
        r = int(lc.diameter / 2 / voxel_size)
        lc_hu.append(np.mean(vol_hu[lc.slice, y_i - r : y_i + r, x_i - r : x_i + r]))

    lcr_hu = sum(lcr_hu) / len(lcr_hu)

    print(lcr_hu)
    print(lc_hu)
    print(list(map(lambda hu: (hu - lcr_hu) / (lcr_hu + hu), lc_hu)))


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    phantom_path = root_path / "phantoms/_2"
    d = 501.7847226
    d_vox = 453
    h = 159.5077264
    phantom_desc = PhantomDesc(
        outer=Cylinder(
            diameter=d,
            height=h,
            mat_name="water",
            mat_value=1.0,
        ),
        inner=[
            InnerCylinder(
                cylinder=Cylinder(
                    diameter=0.1 * d,
                    height=h,
                    mat_name="water",
                    mat_value=1.01,
                ),
                distance=0.3 * d,
                angle=45 * math.pi / 180,
            ),
            InnerCylinder(
                cylinder=Cylinder(
                    diameter=0.1 * d,
                    height=h,
                    mat_name="water",
                    mat_value=1.02,
                ),
                distance=0.3 * d,
                angle=(45 + 90) * math.pi / 180,
            ),
            InnerCylinder(
                cylinder=Cylinder(
                    diameter=0.1 * d,
                    height=h,
                    mat_name="water",
                    mat_value=1.05,
                ),
                distance=0.3 * d,
                angle=(45 + 180) * math.pi / 180,
            ),
            InnerCylinder(
                cylinder=Cylinder(
                    diameter=0.1 * d,
                    height=h,
                    mat_name="water",
                    mat_value=1.10,
                ),
                distance=0.3 * d,
                angle=(45 + 270) * math.pi / 180,
            ),
        ],
    )

    # d_vox, h_vox = make_phantom_by_desc(phantom_desc, d / d_vox, phantom_path)
    h_vox = 144

    lcr = [Region(distance=220, angle=i * 90 * math.pi / 180, diameter=20, slice=h_vox // 2) for i in range(4)]
    lc = [
        Region(distance=150, angle=(45 + i * 90) * math.pi / 180, diameter=20, slice=h_vox // 2)
        for i in range(3, -1, -1)
    ]

    # cfg_path = root_path / "cfg_flat"
    # xcist = make_projections(cfg_path, phantom_path)
    # cfg_str = cfg_to_str(xcist)
    # with open(phantom_path / "xcist.cfg", "w", newline="\n") as f:
    #     f.write(cfg_str)

    xcist = xc.CatSim(str(phantom_path / "xcist.cfg"))
    # recon = make_reconstruct(phantom_path / "projs.tif", xcist, d / d_vox)
    # tifffile.imwrite(phantom_path / "recon.tif", recon, imagej=True, compression="zlib")
    recon = tifffile.imread(phantom_path / "recon.tif")
    test(recon, d / d_vox, xcist, lcr, lc)


if __name__ == "__main__":
    main()
