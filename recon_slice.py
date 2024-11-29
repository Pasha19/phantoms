import functools
import json
import pathlib

import numpy as np
from PIL import Image
import tifffile
import tomosipo as ts
import torch
from tqdm import tqdm
from ts_algorithms import fdk


def cmp_slices(slice1: pathlib.Path, slice2: pathlib.Path) -> int:
    n1 = int(slice1.name.split('_')[1])
    n2 = int(slice2.name.split('_')[1])
    return n1 - n2


def rec(slices_path: pathlib.Path, image_shape: int, voxel_size: float) -> np.ndarray:
    slices = list(slices_path.glob("slice_*"))
    slices = sorted(slices, key=functools.cmp_to_key(cmp_slices))
    volume = np.zeros((144, image_shape, image_shape))
    for i in tqdm(range(len(slices))):
        sl = rec_slice(slices[i], image_shape, voxel_size)
        tifffile.imwrite(slices_path.parent / "recon.tif", sl, imagej=True, compression="zlib")
        # mid = sl[:, sl.shape[1] // 2, :]
        # mid = (mid - mid.min()) / (mid.max() - mid.min())
        # im = Image.fromarray(255 * mid).convert("L")
        # im.save(slices_path.parent / "recon_slice.png")
        volume[:,i,:] = rec_slice(slices[i], image_shape, voxel_size)
        pass
    return volume


def rec_slice(slice_path: pathlib.Path, image_shape: int, voxel_size: float) -> np.ndarray:
    image_shape_3d = [144, 1, image_shape]
    image_size = list(map(lambda x: x*voxel_size, image_shape_3d))
    projs = tifffile.imread(str(slice_path / "projs.tif"))
    projs = projs[np.newaxis, :, :]
    num = projs.shape[0]
    projs_cfg_path = slice_path / "projs.json"
    with open(projs_cfg_path, "r") as f:
        data = json.load(f)
        pixel_size = data["detector"]["row_size"], data["detector"]["col_size"]
        dso = data["sid"]
        dsd = data["sdd"]
        detector_shape = [projs.shape[1], projs.shape[2]]
        detector_size = [detector_shape[0]*pixel_size[0], detector_shape[1]*pixel_size[1]]
        angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    vg = ts.volume(shape=image_shape_3d, size=image_size)
    pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
    A = ts.operator(vg, pg)

    sino = np.transpose(projs, (1, 0, 2))
    sino = torch.from_numpy(sino).cuda()

    recon = fdk(A, sino)
    recon = recon.detach().cpu().numpy()
    return recon


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    phantom_path = root_path / "data2"
    recon = rec(phantom_path / "slices", 453, voxel_size=471/453)
    recon = np.ascontiguousarray(recon.astype(np.float32))
    tifffile.imwrite(phantom_path / "recon.tif", recon, imagej=True, compression="zlib")
    volume = (recon - recon.min()) / (recon.max() - recon.min())
    im = Image.fromarray(255 * volume[:, volume.shape[1] // 2, :]).convert("L")
    im.save(phantom_path / "slice.png")



if __name__ == "__main__":
    main()
