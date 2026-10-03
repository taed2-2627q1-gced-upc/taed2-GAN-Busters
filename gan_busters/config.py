from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Reproducibility
RANDOM_SEED = 1

# Dataset
CIFAR10_CLASSES = {
    1: "airplane",
    2: "automobile",
    3: "bird",
    4: "cat",
    5: "deer",
    6: "dog",
    7: "frog",
    8: "horse",
    9: "ship",
    10: "truck",
}

# Input
INPUT_SHAPE = (32, 32, 3)
NORMALIZE_PIXELS = True

# Base architecture
DEFAULT_CONV_FILTERS = (32, 32)
DEFAULT_DENSE_UNITS = (64,)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_PADDING = "same"
DEFAULT_POOLING = "max"
DEFAULT_POOL_SIZE = (2, 2)

# Default training configuration
DEFAULT_LOSS = "binary_crossentropy"
DEFAULT_OPTIMIZER = "adam"
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT = 0.0

# Phase 1 search space
PHASE1_KERNEL_SIZES = (3, 5)
PHASE1_PADDING = ("valid", "same")
PHASE1_POOLING = ("max", "average")

# Phase 2 search space
PHASE2_LEARNING_RATES = (1e-2, 1e-3, 1e-4)
PHASE2_BATCH_SIZES = (32, 64, 128)
PHASE2_DROPOUT = (0.0, 0.125, 0.25)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
