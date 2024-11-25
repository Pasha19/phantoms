import pathlib

import numpy as np
import tifffile

from make_phantom import make_phantom


def make_box_volume(
        size_zxy_mm: tuple[float, float, float],
        size_box_mm: tuple[float, float, float],
        box_zxy_offset_mm: tuple[float, float, float],
        vox_size_mm: float,
) -> np.ndarray:
    vol_zxy_vox = int(size_zxy_mm[0] / vox_size_mm + 0.5),\
        int(size_zxy_mm[1] / vox_size_mm + 0.5),\
        int(size_zxy_mm[2] / vox_size_mm + 0.5)
    box_zxy_vox = int(size_box_mm[0] / vox_size_mm + 0.5), \
        int(size_box_mm[1] / vox_size_mm + 0.5), \
        int(size_box_mm[2] / vox_size_mm + 0.5)
    box_zxy_offset_vox = int(box_zxy_offset_mm[0] / vox_size_mm + 0.5),\
        int(box_zxy_offset_mm[1] / vox_size_mm + 0.5),\
        int(box_zxy_offset_mm[2] / vox_size_mm + 0.5)

    center_zxy_vox = vol_zxy_vox[0] // 2, vol_zxy_vox[1] // 2, vol_zxy_vox[2] // 2

    z_2, x_2, y_2 = box_zxy_vox[0] // 2, box_zxy_vox[1] // 2, box_zxy_vox[2] // 2

    volume = np.zeros(shape=vol_zxy_vox, dtype=np.float32)
    volume[
        center_zxy_vox[0]+box_zxy_offset_vox[0] - z_2: center_zxy_vox[0]+box_zxy_offset_vox[0] + z_2,
        center_zxy_vox[1] + box_zxy_offset_vox[1] - x_2: center_zxy_vox[1] + box_zxy_offset_vox[1] + x_2,
        center_zxy_vox[2] + box_zxy_offset_vox[2] - y_2: center_zxy_vox[2] + box_zxy_offset_vox[2] + y_2,
    ] = 1

    return volume


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()

    size_zxy_mm = 159.5077264, 501.7847226, 501.7847226/10
    size_box_mm = 100, 400, 400/10
    cyl_zxy_offset_mm = 0, 0, 0
    vox_size_mm = 501.7847226 / 453

    box = make_box_volume(size_zxy_mm, size_box_mm, cyl_zxy_offset_mm, vox_size_mm)
    # tifffile.imwrite('box.tif', box, imagej=True, compression="zlib")
    make_phantom(root_path / "box", box, vox_size_mm, 1)


if __name__ == "__main__":
    main()
