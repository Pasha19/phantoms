from pathlib import Path

import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import numpy as np
from gecatsim.pyfiles.Prep_BHC_Accurate import Prep_BHC_Accurate
from tifffile import tifffile

from make_slice_gt import cfg_to_str
from test_hu import make_reconstruct


def get_xcist(cfg_path: Path, phantom_json_path: Path, projs_path: Path) -> xc.CatSim:
    cfg = [str(file) for file in cfg_path.glob("*.cfg")]

    xcist = xc.CatSim(*cfg)
    xcist.cfg.phantom.filename = str(phantom_json_path)
    xcist.resultsName = str(projs_path)

    my_path = xc.pyfiles.CommonTools.my_path
    my_path.add_search_path(str(cfg_path / "spectrum"))
    my_path.add_search_path(str(cfg_path / "material"))

    return xcist


def do_projs(xcist: xc.CatSim) -> None:
    xcist.run_all()

    num = xcist.cfg.protocol.viewCount
    rows = xcist.cfg.scanner.detectorRowCount
    cols = xcist.cfg.scanner.detectorColCount

    result_path = Path(xcist.resultsName)

    projs = xc.rawread(str(result_path.with_suffix(".prep")), (num, rows, cols), "float")
    tifffile.imwrite(result_path.with_suffix(".tif"), projs, imagej=True, compression="zlib")


def do_xcist_recon(xcist: xc.CatSim) -> None:
    xcist.do_Recon = 1
    # xcist.resultsName = str(Path(xcist.resultsName).parent / name)
    recon.recon(xcist)


def bhc(xcist: xc.CatSim) -> None:
    bhc_reference_energy = 80
    bhc_reference_material = "water_20C"

    mu = xc.GetMu(bhc_reference_material, np.array(bhc_reference_energy, dtype=np.float32))

    cfg = xcist.get_current_cfg()
    cfg.physics.EffectiveMu = mu
    cfg.physics.BHC_poly_order = 5
    cfg.physics.BHC_max_length_mm = 500
    cfg.physics.BHC_length_step_mm = 10

    projs = xc.rawread(
        xcist.resultsName + ".prep",
        (cfg.protocol.viewCount, cfg.scanner.detectorRowCount, cfg.scanner.detectorColCount),
        "float",
    )

    tifffile.imwrite(xcist.resultsName + ".prebh.tif", projs, compression="zlib")
    xc.rawwrite(xcist.resultsName + ".prebh", projs)
    projs = Prep_BHC_Accurate(cfg, projs.reshape(xcist.cfg.protocol.viewCount, -1))

    projs = projs.reshape(cfg.protocol.viewCount, cfg.scanner.detectorRowCount, cfg.scanner.detectorColCount)
    tifffile.imwrite(xcist.resultsName + ".tif", projs, compression="zlib")
    xc.rawwrite(xcist.resultsName + ".prep", projs)


def main() -> None:
    root_path = Path(__file__).parent.resolve()
    # det = "curved"
    det = "flat"
    cfg_path = root_path / f"cfg_button_{det}"
    phantom_json_path = root_path / "button" / "Button_1200.json"
    projs_path = phantom_json_path.parent / f"projs_{det}"

    xcist = get_xcist(cfg_path, phantom_json_path, projs_path)
    with open(phantom_json_path.parent / f"xcist_{det}.cfg", "w", newline="\n") as f:
        f.write(cfg_to_str(xcist))

    # do_projs(xcist)
    # bhc(xcist)

    recon = make_reconstruct(projs_path.with_suffix(".tif"), xcist, 0.009)
    tifffile.imwrite(phantom_json_path.parent / f"recon_tomosipo_{det}.tif", recon, imagej=True, compression="zlib")

    do_xcist_recon(xcist)


if __name__ == "__main__":
    main()
