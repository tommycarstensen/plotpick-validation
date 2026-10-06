"""Shared path configuration for validation scripts.

Large datasets (images, zips) live outside the repository, so that a
synced or backed-up working folder does not carry gigabytes of chart
images. Small metadata (JSON, CSV) stays in the validation/ folder.

Set PLOTPICK_DATA to choose where the large data lives; it defaults to
~/plotpick_data.

Scripts in subdirs (benchmarks/, pipeline/) add VALIDATION_ROOT to
sys.path so they can ``from paths import ...``.
"""

import os
from pathlib import Path

VALIDATION_ROOT = Path(__file__).resolve().parent


def _local_data():
    """Where the large datasets live.

    PLOTPICK_DATA selects the directory; the default is ~/plotpick_data.
    """
    env = os.environ.get("PLOTPICK_DATA")
    return Path(env) if env else Path.home() / "plotpick_data"


LOCAL_DATA = _local_data()

# API key
SECRETS_PATH = VALIDATION_ROOT.parent / "plotpick" / ".streamlit" / "secrets.toml"

# PMC pair metadata
PAIRS_PATH = VALIDATION_ROOT / "pairs.json"
EXCLUDED_PATH = VALIDATION_ROOT / "excluded_pairs.json"
CANDIDATES_PATH = VALIDATION_ROOT / "candidates.csv"

# External benchmark datasets
EXTERNAL_DIR = VALIDATION_ROOT / "external"
CHARTX_DIR = EXTERNAL_DIR / "chartx"
CHARTX_META = CHARTX_DIR / "chartx_test.json"
CHARTX_VAL_META = CHARTX_DIR / "chartx_val.json"
CHARTX_IMG_ROOT = LOCAL_DATA / "datasets" / "chartx_images" / "ChartX_png"

# PlotQA
PLOTQA_DIR = LOCAL_DATA / "datasets" / "plotqa"

# PMC figures & PDFs (large, local only)
PMC_FIG_DIR = LOCAL_DATA / "figures"
PDF_DIR = LOCAL_DATA / "pdfs"

# Tables (small, kept in the repo folder)
TABLE_DIR = VALIDATION_ROOT / "tables"

# Results (organised by benchmark)
RESULTS_DIR = VALIDATION_ROOT / "results"
CHARTX_RESULTS_DIR = RESULTS_DIR / "chartx"
PLOTQA_RESULTS_DIR = RESULTS_DIR / "plotqa"
FINAL_VAL_RESULTS_DIR = RESULTS_DIR / "final_val"
PMC_RESULTS_DIR = RESULTS_DIR / "pmc"
