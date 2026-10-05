"""Download the ChartX chart images that the ChartX benchmarks read.

The annotations (external/chartx/chartx_test.json, chartx_val.json) are in
the repo; the ~440 MB of PNGs are not. This fetches ChartX_png.zip from the
InternScience/ChartX dataset on Hugging Face and unpacks it so that
CHARTX_IMG_ROOT (paths.py) holds <chart_type>/png/<imgname>.png.

Usage:
    python pipeline/fetch_chartx_images.py
"""

import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CHARTX_IMG_ROOT  # noqa: E402

ZIP_URL = ("https://huggingface.co/datasets/InternScience/ChartX/"
           "resolve/main/ChartX_png.zip")


def main():
    if CHARTX_IMG_ROOT.is_dir():
        print(f"Already present: {CHARTX_IMG_ROOT}")
        return

    dest = CHARTX_IMG_ROOT.parent
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ChartX_png.zip"
    part_path = zip_path.with_suffix(".zip.part")

    if not zip_path.exists():
        print(f"Downloading {ZIP_URL}")
        with requests.get(ZIP_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(part_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB",
                              end="", flush=True)
        print()
        part_path.rename(zip_path)

    print(f"Extracting to {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    if not CHARTX_IMG_ROOT.is_dir():
        sys.exit(f"Extraction finished but {CHARTX_IMG_ROOT} does not exist; "
                 "the zip layout may have changed.")
    zip_path.unlink()
    n = sum(1 for _ in CHARTX_IMG_ROOT.glob("*/png/*.png"))
    print(f"Done: {n} images under {CHARTX_IMG_ROOT}")


if __name__ == "__main__":
    main()
