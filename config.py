import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# dotenv constants
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path=dotenv_path)

UNSWNB15_ROOT = Path(os.getenv("UNSWNB15_ROOT"))
CICIDS2017_ROOT = Path(os.getenv("CICIDS2017_ROOT"))
# BOTIOT_ROOT = Path(os.getenv("BOTIOT_ROOT"))

LOG_LEVEL = os.getenv("LOG_LEVEL") or "INFO"

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "figures"
INTERM_DIR = Path(ROOT) / "interm"
DATA_ROOT = Path(ROOT) / "IDS datasets"
...  # others
