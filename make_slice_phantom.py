import json
import pathlib
import shutil

import numpy as np
import pydicom
import tifffile

from FixedResampleVolume import FixedResampleVolume
from make_projections import make_projs


def read_phantom(phantom_dir_path: pathlib.Path) -> tuple[np.ndarray, float, float]:
    slice_thickness_key = 0x18, 0x50
    rows_key = 0x28, 0x10
    cols_key = 0x28, 0x11
    pixel_spacing_key = 0x28, 0x30
    pixel_padding_key = 0x28, 0x0120
    rescale_intercept_key = 0x28, 0x1052
    rescale_slope_key = 0x28, 0x1053

    size = 512

    dcm_files = list(phantom_dir_path.glob('*.dcm'))
    volume = np.empty((len(dcm_files), size, size), dtype=np.float32)
    pixel_spacing_last = None
    slice_thickness_last = None
    for i, path in enumerate(dcm_files):
        dcm = pydicom.dcmread(path)
        rows = int(dcm[rows_key].value)
        cols = int(dcm[cols_key].value)
        assert rows == size and cols == size

        slice_thickness = float(dcm[slice_thickness_key].value)
        pixel_spacing = dcm[pixel_spacing_key].value
        assert pixel_spacing[0] == pixel_spacing[1]

        pixel_spacing = float(pixel_spacing[0])
        rescale_intercept = float(dcm[rescale_intercept_key].value)
        rescale_slope = float(dcm[rescale_slope_key].value)
        img = dcm.pixel_array
        img = np.astype(img, np.float32)
        if pixel_padding_key in dcm:
            pixel_padding = int(dcm[pixel_padding_key].value)
            img[img==pixel_padding] = rescale_intercept
        img[img < 0] = 0
        img *= rescale_slope
        img /= -rescale_intercept

        assert pixel_spacing_last is None or np.isclose(pixel_spacing, pixel_spacing_last)
        assert slice_thickness_last is None or np.isclose(slice_thickness, slice_thickness_last)

        pixel_spacing_last = pixel_spacing
        slice_thickness_last = slice_thickness
        volume[i] = img

    return volume, pixel_spacing_last, slice_thickness_last


def scale_volume(volume: np.ndarray, voxel_size: float, slice_thickness: float) -> tuple[np.ndarray, float]:
    new_voxel_size = voxel_size * volume.shape[1] / 453
    vedo_volume = FixedResampleVolume(volume, spacing=(slice_thickness, voxel_size, voxel_size))
    vedo_volume.resample(new_spacing=[new_voxel_size, new_voxel_size, new_voxel_size], interpolation=1)
    return np.ascontiguousarray(vedo_volume.tonumpy()), new_voxel_size


def create_config(volume: np.ndarray, voxel_size: float, volume_raw_name: str) -> dict:
    size = volume.shape
    cfg = {
        "n_materials": 1,
        "mat_name": ["water"],
        "volumefractionmap_filename": [volume_raw_name],
        "volumefractionmap_datatype": ["float"],
        "rows": [size[1]],
        "cols": [size[2]],
        "slices": [size[0]],
        "x_size": [voxel_size],
        "y_size": [voxel_size],
        "z_size": [voxel_size],
        "x_offset": [size[1]/2],
        "y_offset": [size[2]/2],
        "z_offset": [size[0]/2],
    }
    return cfg


def make_phantom(
        phantom_path: pathlib.Path,
        volume: np.ndarray,
        voxel_size: float,
) -> None:
    phantom_path.mkdir(exist_ok=True, parents=True)
    volume_raw_name = "phantom_water.raw"
    with open(phantom_path / volume_raw_name, "wb") as f:
        f.write(volume)
    cfg = create_config(volume, voxel_size, volume_raw_name)
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)
    tifffile.imwrite(phantom_path / "volume.tif", volume, imagej=True, compression="zlib")


def run(
        phantom_path: pathlib.Path,
        cfg_path: pathlib.Path,
        slice_volume: np.ndarray,
        i: int,
        slices: int,
        voxel_size: float,
        delete_slices: bool,
) -> None:
    slice_projs_path = phantom_path / "slice_projs"
    slice_projs_path.mkdir(exist_ok=True, parents=True)
    slice_path = phantom_path / f"slices/slice_{i:03}"
    # slice_volume = np.pad(slice_volume, ((0, 0), (slices // 2, slices // 2), (0, 0)), mode="constant")
    slice_volume = np.ascontiguousarray(slice_volume)
    make_phantom(slice_path, slice_volume, voxel_size)
    make_projs(slice_path, cfg_path)
    shutil.move(slice_path / "projs.tif", slice_projs_path / f"projs_{i:03}.tif")
    if delete_slices:
        shutil.rmtree(slice_path)


def make_slice_phantoms_with_projection(
        phantom_path: pathlib.Path,
        volume: np.ndarray,
        voxel_size: float,
        cfg_path: pathlib.Path,
        delete_slices: bool,
) -> None:
    slices = volume.shape[2]
    for i in range(slices):
        slice_projs_path = phantom_path / "slice_projs"
        slice_projs_path.mkdir(exist_ok=True, parents=True)
        slice_path = phantom_path / f"slices/slice_{i:03}"
        slice_volume = volume[:, np.newaxis, i]
        slice_volume = np.pad(slice_volume, ((0, 0), (slices // 2 - 1, slices // 2), (0, 0)), mode="constant")
        slice_volume = np.ascontiguousarray(slice_volume)
        make_phantom(slice_path, slice_volume, voxel_size)
        make_projs(slice_path, cfg_path)
        shutil.move(slice_path / "projs.tif", slice_projs_path / f"projs_{i:03}.tif")
        if delete_slices:
            shutil.rmtree(slice_path)
    if delete_slices:
        shutil.rmtree(phantom_path / "slices")


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    img_path = root_path.parent / "LIDC_IDRI/LIDC-IDRI-0011/01-01-2000-NA-NA-73568/3000559.000000-NA-23138"
    volume, voxel_size, slice_thickness = read_phantom(img_path)
    print(f"Before resize max = {volume.max():.2f}; min = {volume.min():.2f}")
    volume, voxel_size = scale_volume(volume, voxel_size, slice_thickness)
    print(f"After resize max = {volume.max():.2f}; min = {volume.min():.2f}")

    d = 501
    h = 150

    volume_d = voxel_size * volume.shape[1]
    scale = d / volume_d

    scaled_voxel_size = scale * voxel_size
    scaled_volume_h = scaled_voxel_size * volume.shape[0]
    keep_slices_num = int(h / scaled_volume_h * volume.shape[0])

    if volume.shape[0] - keep_slices_num > 0:
        start_slice = (volume.shape[0] - keep_slices_num) // 2
        volume = volume[start_slice : start_slice + keep_slices_num]

    phantom_path = root_path.parent / "phantoms/phantom_0011_453_1"
    phantom_path.mkdir(exist_ok=True, parents=True)
    make_phantom(phantom_path, volume, scaled_voxel_size)
    make_slice_phantoms_with_projection(phantom_path, volume, scaled_voxel_size, root_path / "cfg", False)


if __name__ == "__main__":
    main()
