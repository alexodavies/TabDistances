"""Download the TabFM pre-training CSV files from Kaggle.

Prerequisites
-------------
1. Install the Kaggle CLI::

       pip install kaggle

2. Create a Kaggle account, generate an API token from
   https://www.kaggle.com/settings/account, and save it to
   ``~/.kaggle/kaggle.json`` (chmod 600).

Usage
-----
Run from the repository root::

    python download_tabfm.py

Adjust the constants below to control which split to download and where to
store the CSV files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabdistances.data.tabfm.downloader import TabFMDownloader

DATA_PATH = Path("data/tabfm")   # destination for CSV files
SPLIT     = "both"               # "cls", "reg", or "both"

downloader = TabFMDownloader(data_path=DATA_PATH, split=SPLIT)
downloader.download(skip_existing=True, verbose=True)
