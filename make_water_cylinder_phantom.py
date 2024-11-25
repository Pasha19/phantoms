import pathlib

import numpy as np
import tifffile

from make_phantom import make_phantom


def make_cylinder_volume(
        size_zxy_mm: tuple[float, float, float],
        cyl_h_mm: float, cyl_d_mm: float,
        cyl_zxy_offset_mm: tuple[float, float, float],
        vox_size_mm: float,
) -> np.ndarray:
    vol_zxy_vox = int(size_zxy_mm[0] / vox_size_mm + 0.5),\
        int(size_zxy_mm[1] / vox_size_mm + 0.5),\
        int(size_zxy_mm[2] / vox_size_mm + 0.5)
    cyl_h_vox, cyl_d_vox = int(cyl_h_mm / vox_size_mm + 0.5), int(cyl_d_mm / vox_size_mm + 0.5)
    cyl_zxy_offset_vox = int(cyl_zxy_offset_mm[0] / vox_size_mm + 0.5),\
        int(cyl_zxy_offset_mm[1] / vox_size_mm + 0.5),\
        int(cyl_zxy_offset_mm[2] / vox_size_mm + 0.5)

    center_zxy_vox = vol_zxy_vox[0] // 2, vol_zxy_vox[1] // 2, vol_zxy_vox[2] // 2

    volume = np.zeros(shape=vol_zxy_vox, dtype=np.float32)
    z, x, y = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2]),
                          indexing="ij")
    mask_circle = (x - (center_zxy_vox[1] + cyl_zxy_offset_vox[1])) ** 2 + (
                y - (center_zxy_vox[2] + cyl_zxy_offset_vox[2])) ** 2 <= (cyl_d_vox / 2) ** 2
    mask_height = ((center_zxy_vox[0] + cyl_zxy_offset_vox[0]) - (cyl_h_vox / 2) <= z) & (
                z <= (center_zxy_vox[0] + cyl_zxy_offset_vox[0]) + (cyl_h_vox / 2))
    volume[mask_height & mask_circle] = 1

    return volume


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()

    size_zxy_mm = 159.5077264, 501.7847226, 501.7847226
    cyl_h_mm, cyl_d_mm = 100, 400
    cyl_zxy_offset_mm = 0, 0, 0
    vox_size_mm = 501.7847226 / 453

    cyl = make_cylinder_volume(size_zxy_mm, cyl_h_mm, cyl_d_mm, cyl_zxy_offset_mm, vox_size_mm)
    # tifffile.imwrite('cyl.tif', cyl, imagej=True, compression="zlib")
    make_phantom(root_path / "cyl", cyl, vox_size_mm, 1)


if __name__ == "__main__":
    main()
