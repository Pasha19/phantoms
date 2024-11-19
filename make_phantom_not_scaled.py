import pathlib

from make_phantom import make_phantom, read_phantom, scale_volume


def main() -> None:
    root_dir = pathlib.Path(__file__).parent.resolve()
    img_dir = root_dir.parent / r"LIDC_IDRI\LIDC-IDRI-0011\01-01-2000-NA-NA-73568\3000559.000000-NA-23138"
    volume, voxel_size, slice_thickness = read_phantom(img_dir)
    print(f"Before resize max = {volume.max():.2f}; min = {volume.min():.2f}")
    volume = scale_volume(volume, voxel_size, slice_thickness, new_size=453, voxel_equal_sized=False)
    print(f"After resize max = {volume.max():.2f}; min = {volume.min():.2f}")

    d = 501.7847226
    h = 159.5077264

    volume_d = voxel_size * volume.shape[1]
    scale = d / volume_d

    scaled_voxel_size = scale * voxel_size
    scales_slice_thickness = scale * slice_thickness

    scaled_volume_h = scales_slice_thickness * volume.shape[0]
    keep_slices_num = int(h / scaled_volume_h * volume.shape[0] + 0.5)

    if volume.shape[0] - keep_slices_num > 0:
        start_slice = (volume.shape[0] - keep_slices_num) // 2
        volume = volume[start_slice : start_slice + keep_slices_num]

    make_phantom(root_dir.parent / "phantoms/phantom_0011_501_2", volume, scaled_voxel_size, scales_slice_thickness)


if __name__ == "__main__":
    main()
