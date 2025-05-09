import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# dotenv constants
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path=dotenv_path)

DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT:
    DATA_ROOT = Path(DATA_ROOT)
     
LOG_LEVEL = os.getenv("LOG_LEVEL") or "INFO"

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "figures"
INTERM_DIR = Path(ROOT) / "interm"
... # others