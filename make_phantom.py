import json
import pathlib

import numpy as np
import pydicom
import tifffile
import vedo


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
    z_scale_last = None
    for i, path in enumerate(dcm_files):
        dcm = pydicom.dcmread(path)
        rows = int(dcm[rows_key].value)
        cols = int(dcm[cols_key].value)
        assert rows == size and cols == size

        slice_thickness = float(dcm[slice_thickness_key].value)
        pixel_spacing = dcm[pixel_spacing_key].value
        assert pixel_spacing[0] == pixel_spacing[1]

        pixel_spacing = float(pixel_spacing[0])
        pixel_padding = int(dcm[pixel_padding_key].value)
        rescale_intercept = float(dcm[rescale_intercept_key].value)
        rescale_slope = float(dcm[rescale_slope_key].value)

        img = dcm.pixel_array
        img = np.astype(img, np.float32)
        img[img==pixel_padding] = rescale_intercept
        img *= rescale_slope
        img += rescale_intercept

        z_scale = slice_thickness / pixel_spacing
        assert z_scale_last is None or np.isclose(z_scale, z_scale_last)

        z_scale_last = z_scale
        volume[i] = img

    return volume, pixel_spacing, z_scale_last


def scale_volume(volume: np.ndarray, voxel_size: float, z_scale: float, new_side: int) -> tuple[np.ndarray, float]:
    vedo_volume = vedo.Volume(volume, spacing=(voxel_size * z_scale, voxel_size, voxel_size))
    shape = volume.shape
    new_height = int(shape[0] * new_side / shape[1] * z_scale + 0.5)
    vedo_volume.resize([new_height, new_side, new_side])
    return np.ascontiguousarray(vedo_volume.tonumpy()), voxel_size * shape[1] / new_side


def create_config(volume: np.ndarray, voxel_size: float, volume_raw_names: list[str]) -> dict:
    size = volume.shape
    num = 2
    cfg = {
        "n_materials": num,
        "mat_name": ["bone", "water"],
        "volumefractionmap_filename": volume_raw_names[0:num],
        "volumefractionmap_datatype": ["float"] * num,
        "rows": [size[1]] * num,
        "cols": [size[2]] * num,
        "slices": [size[0]] * num,
        "x_size": [voxel_size] * num,
        "y_size": [voxel_size] * num,
        "z_size": [voxel_size] * num,
        "x_offset": [size[1]/2] * num,
        "y_offset": [size[2]/2] * num,
        "z_offset": [size[0]/2] * num,
    }
    return cfg


def make_phantom(
        phantom_path: pathlib.Path,
        volume: np.ndarray,
        voxel_size: float,
        bone_threshold: float,
        water_threshold: float,
        air_threshold: float,
) -> None:
    phantom_path.mkdir(exist_ok=True, parents=True)
    volume_raw_names = ["phantom_bone.raw", "phantom_water.raw"]
    volumes = [
        np.clip((volume - water_threshold) / (bone_threshold - water_threshold), 0, 1),
        np.clip((volume - bone_threshold) / (water_threshold - bone_threshold), 0, 1),
    ]
    volumes[1] = np.where(volume < air_threshold, np.float32(0), volumes[1])
    for vol, volume_name in zip(volumes, volume_raw_names):
        with open(phantom_path / volume_name, "wb") as f:
            f.write(vol)
    cfg = create_config(volume, voxel_size, volume_raw_names)
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)
    tifffile.imwrite(phantom_path / "volume.tif", volume, imagej=True, compression="zlib")


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    img_dir = root_dir / "img"
    volume, voxel_size, z_scale = read_phantom(img_dir)
    print(f"Before resize max HU = {volume.max():.2f}")
    volume, voxel_size = scale_volume(volume, voxel_size, z_scale, 512)
    print(f"After resize max HU = {volume.max():.2f}")

    d = 471  # flat
    h = 160
    # d = 502  # curved
    # h = 100

    volume_d = voxel_size * volume.shape[1]
    scale = d / volume_d

    scaled_voxel_size = scale * voxel_size
    scaled_volume_h = scaled_voxel_size * volume.shape[0]
    keep_slices_num = int(h / scaled_volume_h * volume.shape[0])

    if volume.shape[0] - keep_slices_num > 0:
        start_slice = (volume.shape[0] - keep_slices_num) // 2
        volume = volume[start_slice : start_slice + keep_slices_num]

    make_phantom(root_dir / "phantom0", volume, scaled_voxel_size, 1500, 100, -900)


if __name__ == "__main__":
    main()
