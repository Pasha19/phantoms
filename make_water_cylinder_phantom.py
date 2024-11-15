import json
import pathlib

import numpy as np
import tifffile
import vedo


def make_volume(
        diameter: float,
        height: float,
        side_voxels: int,
        water_hu: float,
        air_hu: float,
        res: int,
) -> tuple[np.ndarray, float]:
    voxel_size = diameter / side_voxels
    cylinder = vedo.Cylinder(r=diameter/2, height=height, res=res)
    volume = cylinder.binarize((1, 0), (voxel_size, voxel_size, voxel_size)).tonumpy()
    volume = np.asarray(volume, np.float32)
    volume = np.transpose(volume, axes=(2, 0, 1))
    volume = np.where(volume == 1, np.float32(water_hu), np.float32(air_hu))
    return np.ascontiguousarray(volume), voxel_size


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


def make_phantom(phantom_path: pathlib.Path, volume: np.ndarray, voxel_size: float, water_hu: float) -> None:
    phantom_path.mkdir(exist_ok=True, parents=True)
    volume_raw_name = "phantom_water.raw"
    vol = (volume == water_hu) * np.float32(1)
    with open(phantom_path / volume_raw_name, "wb") as f:
        f.write(vol)
    cfg = create_config(volume, voxel_size, volume_raw_name)
    with open(phantom_path / "phantom.json", "w", newline="\n") as f:
        json.dump(cfg, f, indent=4)
    tifffile.imwrite(phantom_path / "volume.tif", volume, imagej=True, compression="zlib")


def main() -> None:
    air_hu = -1000
    water_hu = 0

    d = 400
    h = 150
    side_voxels = 512

    volume, voxel_size = make_volume(d, h, side_voxels, water_hu, air_hu, 32)

    root_path = pathlib.Path(__file__).parent.resolve()
    phantom_path = root_path / "phantom_water"
    make_phantom(phantom_path, volume, voxel_size, water_hu)


if __name__ == "__main__":
    main()
