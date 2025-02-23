import json
import math
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from multiprocessing import Pool

import gecatsim as xc
import gecatsim.pyfiles.CommonTools
import numpy as np
import psutil
import tifffile
from gecatsim.pyfiles.GetMu import GetMu
from scipy import ndimage
from tqdm import tqdm


def cfg_param_val_to_str(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value)


def cfg_to_str(ex: xc.CatSim) -> str:
    cfg = ""
    for attr in ex.attrList:
        if hasattr(ex, attr):
            attr_val = getattr(ex, attr)
            if isinstance(
                attr_val,
                (
                    gecatsim.pyfiles.CommonTools.emptyCFG,
                    gecatsim.pyfiles.CommonTools.CFG,
                ),
            ):
                attr_vars = vars(attr_val)
                for attr_var_k, attr_var_v in attr_vars.items():
                    cfg += f"{attr}.{attr_var_k} = {cfg_param_val_to_str(attr_var_v)}\n"
            else:
                cfg += f"{attr} = {cfg_param_val_to_str(attr_val)}\n"
    return cfg


def create_config(
    dims: np.ndarray,
    voxel_size: tuple[float, float, float],
    volume_raw_names: list[str],
    materials: list[str],
    energy: float = 60,
) -> dict:
    num = len(volume_raw_names)
    mu_values = []
    for material in materials:
        # Calculate mu values for each material.
        mu_values.append(GetMu(material, energy)[0])
    cfg = {
        "n_materials": num,
        "mat_name": materials,
        "energy": energy,
        "mu_values": mu_values,
        # "mu_thresholds": mu_values,
        "volumefractionmap_filename": volume_raw_names,
        "volumefractionmap_datatype": ["float"] * num,
        "rows": [dims[1]] * num,
        "cols": [dims[2]] * num,
        "slices": [dims[0]] * num,
        "x_size": [voxel_size[1]] * num,
        "y_size": [voxel_size[2]] * num,
        "z_size": [voxel_size[0]] * num,
        "x_offset": [1 + (dims[2] - 1) / 2] * num,
        "y_offset": [1 + (dims[1] - 1) / 2] * num,
        "z_offset": [1 + (dims[0] - 1) / 2] * num,
    }
    return cfg


def make_tmp_phantom(
    phantom_root_path: pathlib.Path,
    volumes: list[np.ndarray],
    materials: list[str],
    voxel_size: float,
    thin_scale: float,
) -> pathlib.Path:
    phantom_file_names = []
    tmp_id = f"{os.getpid()}"
    for i in range(0, len(materials)):
        phantom_name = f"{tmp_id}.{materials[i]}"
        phantom_file_name = phantom_name + ".raw"
        phantom_file_names.append(phantom_file_name)
        phantom_file_path = phantom_root_path / phantom_file_name
        volume = volumes[i]
        with open(phantom_file_path, "wb") as f:
            f.write(volume)
    cfg = create_config(
        volumes[0].shape,
        (voxel_size, voxel_size, thin_scale * voxel_size),
        phantom_file_names,
        materials,
    )
    phantom_json_path = phantom_root_path / f"{tmp_id}.json"
    with open(phantom_json_path, "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)
    return phantom_json_path


def gen_geometry_in_slice_corection_matrix(
    shape: tuple, voxel_size: float, sdd: float, thin_scale: float
) -> np.ndarray:
    """
    create conus geometry in thin slice pixle correction matrix

    shape - (1, heigth, width)
    """
    slice_scale = np.zeros(shape=shape, dtype=np.float32)
    d = sdd - thin_scale * voxel_size / 2.0
    for i_row in range(0, slice_scale.shape[1]):
        for i_col in range(0, slice_scale.shape[2]):
            x = (i_col + 0.5 - slice_scale.shape[2] / 2.0) * voxel_size
            y = (i_row + 0.5 - slice_scale.shape[1] / 2.0) * voxel_size
            slice_scale[0, i_row, i_col] = d / math.sqrt(d * d + x * x + y * y)
    # i_row, i_col = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    # slice_scale[0] = d / np.sqrt(
    #     d * d
    #     + np.square((i_col + 0.5 - slice_scale.shape[2] / 2) * voxel_size)
    #     + np.square((i_row + 0.5 - slice_scale.shape[1] / 2) * voxel_size)
    # )
    return slice_scale


def resample_volume(volume: np.ndarray, voxel_size: tuple[float, float, float], new_voxel_size) -> np.ndarray:
    zoom = list([voxel_size[i] / new_voxel_size for i in range(3)])
    volume_resampled = ndimage.zoom(volume, zoom=zoom, grid_mode=True, mode="grid-constant", order=1, prefilter=False)
    return volume_resampled


@dataclass
class SliceProjWorkerArgs:
    xcist_tmp_path: pathlib.Path
    slice_ind: int
    volumes: list[np.ndarray]
    phantom_cfg: dict
    voxel_size: float
    thin_scale: float


def slice_proj_worker(args: SliceProjWorkerArgs) -> tuple[np.ndarray, int]:
    slice_phantom_cfg_path = make_tmp_phantom(
        args.xcist_tmp_path.parent,
        args.volumes,
        args.phantom_cfg["mat_name"],
        args.voxel_size,
        args.thin_scale,
    )
    xcist = xc.CatSim(args.xcist_tmp_path)
    sid = xcist.cfg.scanner.sid
    sdd = xcist.cfg.scanner.sdd
    xcist.cfg.phantom.centerOffset = [
        0.0,
        sid - sdd - args.thin_scale * args.voxel_size / 2,
        0.0,
    ]
    xcist.cfg.phantom.filename = str(slice_phantom_cfg_path)
    xcist.resultsName = str(args.xcist_tmp_path.parent / f"slice_proj.{os.getpid()}")
    xcist.run_all()
    path_prep = xcist.resultsName + ".prep"
    projs = xc.rawread(
        path_prep,
        (
            xcist.cfg.protocol.viewCount,
            xcist.cfg.scanner.detectorRowCount,
            xcist.cfg.scanner.detectorColCount,
        ),
        "float",
    )
    # slice_scale = gen_geometry_in_slice_corection_matrix(
    #     projs.shape, args.voxel_size, xcist.cfg.scanner.sdd, args.thin_scale
    # )
    # projs *= slice_scale / args.voxel_size
    return projs, args.slice_ind


def set_stdout_to_devnull() -> None:
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), sys.stdout.fileno())
    sys.stdout = devnull
    sys.stderr = devnull


def create_gt_sliced_thin(ex: xc.CatSim, thin_scale: float = 0.05) -> np.ndarray:
    phantom_cfg_json = pathlib.Path(ex.cfg.phantom.filename)
    with open(phantom_cfg_json, "r") as f:
        phantom_cfg = json.load(f)
    volumes = []
    det_pix_size = ex.cfg.scanner.detectorRowSize
    sid = ex.cfg.scanner.sid
    sdd = ex.cfg.scanner.sdd
    voxel_size = det_pix_size / (sdd / sid)
    for i in range(phantom_cfg["n_materials"]):
        volume_file = str(phantom_cfg_json.parent / phantom_cfg["volumefractionmap_filename"][i])
        rows, cols, slices = (
            phantom_cfg["rows"][i],
            phantom_cfg["cols"][i],
            phantom_cfg["slices"][i],
        )
        volume = xc.rawread(volume_file, (slices, rows, cols), "float")
        # im = Image.fromarray(255 * volume[ volume.shape[1] // 2, :, :]).convert("L")
        # im.save(phantom_cfg_json.parent / f"{phantom_cfg["volumefractionmap_filename"][i]}.{phantom_cfg["mat_name"][i]}.png")
        x_size, y_size, z_size = (
            phantom_cfg["x_size"][i],
            phantom_cfg["y_size"][i],
            phantom_cfg["z_size"][i],
        )
        if y_size != voxel_size:
            volume = resample_volume(volume, (x_size, y_size, z_size), voxel_size)
        volumes.append(volume)
        new_size_in_vox = volume.shape[1]
    tmp_path = phantom_cfg_json.parent / f"tmp.{os.getpid()}"
    tmp_path.mkdir(exist_ok=True, parents=True)
    vol_gt = np.zeros(
        (
            ex.cfg.scanner.detectorRowCount,
            ex.cfg.scanner.detectorColCount,
            ex.cfg.scanner.detectorColCount,
        ),
        dtype=np.float32,
    )
    ex.cfg.protocol.viewCount = 1
    ex.cfg.protocol.stopViewId = 0
    xcist_tmp_path = tmp_path / "xcist.cfg"
    with open(xcist_tmp_path, "w") as f:
        f.write(cfg_to_str(ex))
    process_count = psutil.cpu_count(logical=False)
    with Pool(processes=process_count, initializer=set_stdout_to_devnull) as pool:
        start_ind = max(0, (new_size_in_vox - ex.cfg.scanner.detectorColCount) // 2)
        end_ind = min(ex.cfg.scanner.detectorColCount, start_ind + new_size_in_vox)
        offset_ind = max(0, (ex.cfg.scanner.detectorColCount - new_size_in_vox) // 2)
        args = [
            SliceProjWorkerArgs(
                xcist_tmp_path=xcist_tmp_path,
                slice_ind=offset_ind + i,
                volumes=list(vol[:, start_ind + i : start_ind + i + 1, :] for vol in volumes),
                phantom_cfg=phantom_cfg,
                voxel_size=det_pix_size,
                thin_scale=thin_scale,
            )
            for i in range(end_ind - start_ind)
        ]
        for projs, slice_ind in tqdm(pool.imap_unordered(slice_proj_worker, args), total=len(args)):
            vol_gt[:, slice_ind : slice_ind + 1, :] = np.transpose(projs, (1, 0, 2))
    slice_scale = gen_geometry_in_slice_corection_matrix(projs.shape, det_pix_size, ex.cfg.scanner.sdd, thin_scale)
    slice_scale = np.transpose(slice_scale, (1, 0, 2))
    vol_gt *= slice_scale / det_pix_size
    shutil.rmtree(tmp_path)
    return vol_gt


def xcist_from_config_path(cfg_path: pathlib.Path) -> xc.CatSim:
    if cfg_path.is_dir():
        return xc.CatSim(*list(map(str, cfg_path.glob("*.cfg"))))
    return xc.CatSim(str(cfg_path))


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg_path = root_path / "cfg_flat"
    xcist = xcist_from_config_path(cfg_path)
    phantom_path = root_path / "phantoms/_test" / "phantom.json"
    xcist.cfg.phantom.filename = str(phantom_path)
    vol_gt = create_gt_sliced_thin(xcist)
    tifffile.imwrite(phantom_path.parent / "phantom.gt.tif", vol_gt, compression="zlib")


if __name__ == "__main__":
    main()
