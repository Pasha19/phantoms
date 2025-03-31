import json
import pathlib

import gecatsim as xc
import tifffile


def do_projs(phantom_json_path: pathlib.Path, cfg: list[str], projs_path: pathlib.Path) -> xc.CatSim:
    projs_path.mkdir(parents=True, exist_ok=True)
    xcist = xc.CatSim(*cfg)
    xcist.cfg.phantom.filename = str(phantom_json_path)
    xcist.resultsName = str(projs_path / "projs")
    xcist.run_all()
    return xcist


def get_config(cfg_path: pathlib.Path) -> list[str]:
    return [
        str(cfg_path / "phantom.cfg"),
        str(cfg_path / "physics.cfg"),
        str(cfg_path / "protocol.cfg"),
        str(cfg_path / "scanner.cfg"),
    ]


def make_proj_config(xcist: xc.CatSim) -> dict:
    return {
        "detector": {
            "row_size": xcist.cfg.scanner.detectorRowSize,
            "col_size": xcist.cfg.scanner.detectorColSize,
        },
        "sid": xcist.cfg.scanner.sid,
        "sdd": xcist.cfg.scanner.sdd,
    }


def main() -> None:
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg_path = root_path / "cfg_flat"
    phantom_path = root_path / "phantoms/_1/"
    phantom_json_path = phantom_path / "phantom.json"
    xcist = do_projs(
        phantom_json_path,
        get_config(cfg_path),
        phantom_path,
    )

    num = xcist.cfg.protocol.viewCount
    rows = xcist.cfg.scanner.detectorRowCount
    cols = xcist.cfg.scanner.detectorColCount
    projs = xc.rawread(str(phantom_path / "projs.prep"), (num, rows, cols), "float")
    tifffile.imwrite(phantom_path / f"projs.tif", projs, imagej=True, compression="zlib")
    with open(phantom_path / "projs.json", "w", newline="\n") as f:
        json.dump(make_proj_config(xcist), f, indent=4)


if __name__ == "__main__":
    main()


def get_xcist(phantom_json_path: pathlib.Path, cfg: list[str], projs_path: pathlib.Path, projs_name: str) -> xc.CatSim:
    projs_path.mkdir(parents=True, exist_ok=True)
    xcist = xc.CatSim(*cfg)
    xcist.cfg.phantom.filename = str(phantom_json_path)
    xcist.resultsName = str(projs_path / projs_name)
    return xcist


def make_projs(phantom_path: pathlib.Path, cfg_path: pathlib.Path) -> None:
    phantom_json_path = phantom_path / "phantom.json"
    projs_name = "projs2"
    xcist = get_xcist(phantom_json_path, get_config(cfg_path), phantom_path, projs_name)
    xcist.run_all()
    num = xcist.cfg.protocol.viewCount
    rows = xcist.cfg.scanner.detectorRowCount
    cols = xcist.cfg.scanner.detectorColCount
    projs = xc.rawread(str(phantom_path / f"{projs_name}.prep"), (num, rows, cols), "float")
    tifffile.imwrite(phantom_path / f"{projs_name}.tif", projs, imagej=True, compression="zlib")
    with open(phantom_path / f"{projs_name}.json", "w", newline="\n") as f:
        json.dump(make_proj_config(xcist), f, indent=4)
