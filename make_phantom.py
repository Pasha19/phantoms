import json
import pathlib

import numpy as np
import pydicom
import tifffile

from FixedResampleVolume import FixedResampleVolume


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


def scale_volume(
        volume: np.ndarray,
        voxel_size: float,
        slice_thickness: float,
        new_size: int,
        voxel_equal_sized: bool,
) -> np.ndarray:
    vedo_volume = FixedResampleVolume(volume, spacing=(slice_thickness, voxel_size, voxel_size))
    new_voxel_size = voxel_size * volume.shape[1] / new_size
    if voxel_equal_sized:
        vedo_volume.resample(new_spacing=[new_voxel_size, new_voxel_size, new_voxel_size], interpolation=1)
    else:
        new_voxel_slice_size = slice_thickness * volume.shape[1] / new_size
        vedo_volume.resample(new_spacing=[new_voxel_slice_size, new_voxel_size, new_voxel_size])
    return np.ascontiguousarray(vedo_volume.tonumpy())


def create_config(
        volume: np.ndarray,
        voxel_size: tuple[float, float, float],
        volume_raw_name: str,
) -> dict:
    size = volume.shape
    cfg = {
        "n_materials": 1,
        "mat_name": ["water"],
        "volumefractionmap_filename": [volume_raw_name],
        "volumefractionmap_datatype": ["float"],
        "rows": [size[1]],
        "cols": [size[2]],
        "slices": [size[0]],
        "x_size": [voxel_size[1]],
        "y_size": [voxel_size[2]],
        "z_size": [voxel_size[0]],
        "x_offset": [size[1]/2],
        "y_offset": [size[2]/2],
        "z_offset": [size[0]/2],
    }
    return cfg


def make_phantom(
        phantom_path: pathlib.Path,
        volume: np.ndarray,
        voxel_size: float,
        slice_thickness: float,
) -> None:
    phantom_path.mkdir(exist_ok=True, parents=True)
    volume_raw_name = "phantom_water.raw"
    with open(phantom_path / volume_raw_name, "wb") as f:
        f.write(volume)
    cfg = create_config(volume, (slice_thickness, voxel_size, voxel_size), volume_raw_name)
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)
    tifffile.imwrite(phantom_path / "volume.tif", volume, imagej=True, compression="zlib")


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    img_dir = root_dir.parent / r"LIDC_IDRI\LIDC-IDRI-0011\01-01-2000-NA-NA-73568\3000559.000000-NA-23138"
    volume, voxel_size, slice_thickness = read_phantom(img_dir)
    print(f"Before resize max = {volume.max():.2f}; min = {volume.min():.2f}")
    volume = scale_volume(volume, voxel_size, slice_thickness, new_size=453, voxel_equal_sized=True)
    print(f"After resize max = {volume.max():.2f}; min = {volume.min():.2f}")

    d = 501.7847226
    h = 159.5077264

    volume_d = voxel_size * volume.shape[1]
    scale = d / volume_d

    scaled_voxel_size = scale * voxel_size
    scaled_volume_h = scaled_voxel_size * volume.shape[0]
    keep_slices_num = int(h / scaled_volume_h * volume.shape[0])

    if volume.shape[0] - keep_slices_num > 0:
        start_slice = (volume.shape[0] - keep_slices_num) // 2
        volume = volume[start_slice : start_slice + keep_slices_num]

    make_phantom(root_dir.parent / "phantoms/phantom_0011_501", volume, scaled_voxel_size, scaled_voxel_size)


if __name__ == "__main__":
    main()
