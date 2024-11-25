import json
import pathlib

import numpy as np
import pydicom
from skimage import measure, morphology
import tifffile

from FixedResampleVolume import FixedResampleVolume


def read_phantom(phantom_dir_path: pathlib.Path) -> tuple[np.ndarray, float, float, float]:
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
    rescale_intercept = 0.0
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
        img *= rescale_slope
        img += rescale_intercept

        assert pixel_spacing_last is None or np.isclose(pixel_spacing, pixel_spacing_last)
        assert slice_thickness_last is None or np.isclose(slice_thickness, slice_thickness_last)

        pixel_spacing_last = pixel_spacing
        slice_thickness_last = slice_thickness
        volume[i] = img

    return volume, pixel_spacing_last, slice_thickness_last, rescale_intercept


def make_air_mask(volume: np.ndarray, threshold: float) -> np.ndarray:
    # labels = measure.label(morphology.binary_opening(volume < threshold))
    labels = measure.label(volume < threshold)
    label, count = np.unique(labels, return_counts=True)
    label_num = np.argmax(count[1:]) + 1
    mask = np.where(labels == label[label_num], np.uint8(255), np.uint8(0))

    return mask


def resampled_volume(
        volume: np.ndarray,
        voxel_size: float,
        slice_thickness: float,
        new_size: int,
) -> tuple[np.ndarray, float]:
    vedo_volume = FixedResampleVolume(volume, spacing=(slice_thickness, voxel_size, voxel_size))
    new_spacing = float(vedo_volume.ybounds()[1] / (new_size - 1))
    vedo_volume.resample(new_spacing=[new_spacing, new_spacing, new_spacing], interpolation=1)
    return np.ascontiguousarray(vedo_volume.tonumpy()), voxel_size * volume.shape[1] / new_size


def create_config(
        dims: np.ndarray,
        voxel_size: tuple[float, float, float],
        volume_raw_names: list[str],
        materials: list[str],
) -> dict:
    num = len(volume_raw_names)
    cfg = {
        "n_materials": num,
        "mat_name": materials,
        "volumefractionmap_filename": volume_raw_names,
        "volumefractionmap_datatype": ["float"] * num,
        "rows": [dims[1]] * num,
        "cols": [dims[2]] * num,
        "slices": [dims[0]] * num,
        "x_size": [voxel_size[1]] * num,
        "y_size": [voxel_size[2]] * num,
        "z_size": [voxel_size[0]] * num,
        "x_offset": [dims[1]/2] * num,
        "y_offset": [dims[2]/2] * num,
        "z_offset": [dims[0]/2] * num,
    }
    return cfg


def make_phantom(
        phantom_path: pathlib.Path,
        volume: np.ndarray,
        voxel_size: float,
        scale: float,
) -> None:
    phantom_path.mkdir(exist_ok=True, parents=True)
    with open(phantom_path / "phantom_water.raw", "wb") as f:
        f.write(volume)
    scaled_voxel_size = voxel_size * scale
    cfg = create_config(
        volume.shape,
        (scaled_voxel_size, scaled_voxel_size, scaled_voxel_size),
        ["phantom_water.raw"],
        ["water"],
    )
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)


def cut_from_middle(volume: np.ndarray, height: float, voxel_size: float, scale: float):
    scaled_voxel_size = scale * voxel_size
    scaled_volume_height = scaled_voxel_size * volume.shape[0]
    keep_slices_num = int(height / scaled_volume_height * volume.shape[0])

    if volume.shape[0] - keep_slices_num > 0:
        start_slice = (volume.shape[0] - keep_slices_num) // 2
        return volume[start_slice: start_slice + keep_slices_num]
    return volume


def main() -> None:
    air_hu = -975.0
    bone_hu = 1500.0

    downscaled_size = 453
    upscaled_size = 2 * downscaled_size

    d = 501.7847226
    h = 159.5077264

    root_path = pathlib.Path(__file__).parent.resolve()
    img_path = root_path.parent / r"LIDC_IDRI\LIDC-IDRI-0011\01-01-2000-NA-NA-73568\3000559.000000-NA-23138"
    phantom_path = root_path.parent / "phantoms/phantom_0011"
    phantom_path.mkdir(exist_ok=True, parents=True)

    volume, voxel_size, slice_thickness, rescale_intercept = read_phantom(img_path)
    tifffile.imwrite(phantom_path / "volume.tif", volume, imagej=True, compression="zlib")
    volume_d = voxel_size * volume.shape[1]

    volume[volume > bone_hu] = bone_hu
    air_mask = make_air_mask(volume, air_hu)
    tifffile.imwrite(phantom_path / "air_mask.tif", air_mask, compression="zlib")
    volume[air_mask > 0] = air_hu
    tifffile.imwrite(phantom_path / "volume_clipped.tif", volume, imagej=True, compression="zlib")

    volume -= rescale_intercept
    volume /= -rescale_intercept
    volume[air_mask > 0] = 0.0
    tifffile.imwrite(phantom_path / "volume_to_water.tif", volume, imagej=True, compression="zlib")

    volume, voxel_size = resampled_volume(volume, voxel_size, slice_thickness, new_size=upscaled_size)
    if volume.shape[0] % 2 == 1:
        volume = volume[:-1,]
    # volume_upscaled += np.random.uniform(-1, 1, volume_upscaled.shape)
    tifffile.imwrite(phantom_path / "volume_upscaled.tif", volume, imagej=True, compression="zlib")

    scale = d / volume_d

    make_phantom(phantom_path / "upscaled", volume, voxel_size, scale)

    volume, voxel_size = resampled_volume(volume, voxel_size, voxel_size, new_size=downscaled_size)
    tifffile.imwrite(phantom_path / "volume_downscaled.tif", volume, imagej=True, compression="zlib")

    make_phantom(phantom_path / "downscaled", volume, voxel_size, scale)


if __name__ == "__main__":
    main()
