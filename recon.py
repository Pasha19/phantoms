import json
import pathlib

import numpy as np
from PIL import Image
import tifffile
import tomosipo as ts
import torch
from ts_algorithms import fdk


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    phantom_path = root_path.parent / "phantoms/phantom_0011/downscaled"

    voxel_size = 1.107692544
    image_shape = [144, 453, 453]
    image_size = list(map(lambda x: x*voxel_size, image_shape))

    projs = tifffile.imread(str(phantom_path / "projs.tif"))
    num = projs.shape[0]
    projs_cfg_path = phantom_path / "projs.json"
    with open(projs_cfg_path, "r") as f:
        data = json.load(f)
        pixel_size = data["detector"]["row_size"], data["detector"]["col_size"]
        dso = data["sid"]
        dsd = data["sdd"]
    detector_shape = [projs.shape[1], projs.shape[2]]
    detector_size = [detector_shape[0]*pixel_size[0], detector_shape[1]*pixel_size[1]]
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)

    vg = ts.volume(shape=image_shape, size=image_size)
    pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
    A = ts.operator(vg, pg)

    sino = np.transpose(projs, (1, 0, 2))
    sino = torch.from_numpy(sino).cuda()

    recon = fdk(A, sino)
    recon = recon.detach().cpu().numpy()
    tifffile.imwrite(phantom_path / "recon.tif", recon, imagej=True, compression="zlib")
    volume = (recon - recon.min()) / (recon.max() - recon.min())
    im = Image.fromarray(255 * volume[:, volume.shape[1] // 2, :]).convert("L")
    im.save(phantom_path / "slice.png")

if __name__ == "__main__":
    main()
