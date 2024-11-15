import json
import pathlib

import numpy as np
from PIL import Image
import tifffile


def make_phantom(
        volume_d: float,
        volume_h: float,
        pad_h: float,
        volume_side: int,
        step: tuple[float, float],
        sub_steps: tuple[int, int],
) -> np.ndarray:
    step_horizontal, step_vertical = step
    sub_steps_horizontal, sub_steps_vertical = sub_steps

    total_h = volume_h + 2*pad_h
    voxel_height = int(volume_side / volume_d * total_h + 0.5)
    volume = np.zeros((voxel_height, volume_side, volume_side), dtype=np.float32)

    x_center = volume_side // 2
    y_center = volume_side // 2
    z_center = voxel_height // 2

    volume_h_voxels = int(volume_side / volume_d * volume_h + 0.5)
    volume_h_start = (voxel_height - volume_h_voxels) // 2
    volume[volume_h_start : volume_h_start + volume_h_voxels, x_center, (0, -1)] = 1
    volume[(volume_h_start, volume_h_start + volume_h_voxels), x_center, :] = 1

    height = int(0.2 * voxel_height)
    height_start = (voxel_height - height) // 2
    small_height = height // 2
    small_height_start = (voxel_height - small_height) // 2
    center_height = 2 * height
    center_height_start = (voxel_height - center_height) // 2

    volume[center_height_start: center_height_start + center_height, x_center, y_center] = 1
    amount = int(sub_steps_horizontal * (volume_d / 2) / step_horizontal) + 1

    last_dist = None
    for i in range(1, amount):
        dist = int(volume_side / volume_d * step_horizontal / sub_steps_horizontal * i)
        if y_center - dist < 0 or y_center + dist >= volume_side:
            break
        if i % sub_steps_horizontal == 0:
            height_ind = height_start, height_start + height
            last_dist = dist
        else:
            height_ind = small_height_start, small_height_start + small_height
        volume[height_ind[0]: height_ind[1], x_center, (y_center - dist, y_center + dist)] = 1

    width = height // 2
    width_start = int((volume_side - width) / 2 + 0.5)
    small_width = width // 2
    small_width_start = int((volume_side - small_width) / 2 + 0.5)
    center_width = 2 * width
    center_width_start = int((volume_side - center_width) / 2 + 0.5)

    volume[z_center, x_center, center_width_start: center_width_start + center_width] = 1
    if last_dist is not None:
        volume[z_center, x_center, center_width_start - last_dist: center_width_start + center_width - last_dist] = 1
        volume[z_center, x_center, center_width_start + last_dist: center_width_start + center_width + last_dist] = 1
    amount = int(sub_steps_vertical * (total_h / 2) / step_vertical) + 1
    for i in range(1, amount):
        dist = int(voxel_height / total_h * step_vertical / sub_steps_vertical * i)
        if z_center - dist < 0 or z_center + dist >= voxel_height:
            break
        if i % sub_steps_vertical == 0:
            y_ind = width_start, width_start + width
        else:
            y_ind = small_width_start, small_width_start + small_width
        volume[(z_center - dist, z_center + dist), x_center, y_ind[0] : y_ind[1]] = 1
        if last_dist is not None:
            volume[(z_center - dist, z_center + dist), x_center, y_ind[0] - last_dist: y_ind[1] - last_dist] = 1
            volume[(z_center - dist, z_center + dist), x_center, y_ind[0] + last_dist: y_ind[1] + last_dist] = 1

    voxel_pad_h = int(volume_side / volume_d * pad_h + 0.5)
    volume[voxel_pad_h + 5 : voxel_pad_h + 10, x_center, 5:10] = 1

    return volume


def create_config(volume: np.ndarray, voxel_size: float, volume_raw_name: str) -> dict:
    size = volume.shape
    return {
        "n_materials": 1,
        "mat_name": ["bone"],
        "volumefractionmap_filename": [volume_raw_name],
        "volumefractionmap_datatype": ["float"],
        "rows": [size[2]],
        "cols": [size[1]],
        "slices": [size[0]],
        "x_size": [voxel_size],
        "y_size": [voxel_size],
        "z_size": [voxel_size],
        "x_offset": [size[2]/2],
        "y_offset": [size[1]/2],
        "z_offset": [size[0]/2],
    }


def write_phantom(volume: np.ndarray, config: dict, path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "phantom.json"
    with open(json_path, "w", newline="\n") as f:
        json.dump(config, f, indent=4)
    volume_path = path / config["volumefractionmap_filename"][0]
    with open(volume_path, "wb") as f:
        f.write(volume)
    tifffile.imwrite(path / "volume.tif", volume, imagej=True, compression="zlib")
    im = Image.fromarray(255 * volume[:, volume.shape[1]//2, :]).convert("L")
    im.save(path / "line.png")


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()

    # d = 471  # flat
    # h = 160
    # d = 502  # curved
    # h = 160
    # size = 512

    # icassp2024
    d = 300
    h = 300
    size = 256

    voxel_size = d / size

    # volume = make_phantom(d, h, 10, size, (50, 20), (4, 4))
    volume = make_phantom(d, h, 10, size, (50, 50), (4, 4))

    config = create_config(volume, voxel_size, "material0.raw")
    write_phantom(volume, config, root_dir / "phantom_line_icassp2024")


if __name__ == "__main__":
    main()
