"""
Central configuration for the GAN-Busters project.

Defines project-wide paths, reproducibility settings, dataset constants,
model/training defaults, and experiment-tracking configuration.

All filesystem paths are resolved from PROJ_ROOT and therefore do not depend
on the current working directory.
"""

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
CORRUPTED_DATA_DIR = DATA_DIR / "corrupted"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ACCEPTED_RECORDS_PATH = PROCESSED_DATA_DIR / "accepted_records.csv"
VALIDATION_RECORDS_PATH = PROCESSED_DATA_DIR / "validation_records.csv"

MODELS_DIR = PROJ_ROOT / "models"
DEFAULT_MODEL_NAME = "gan_busters.keras"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# DagsHub / MLflow
DAGSHUB_OWNER = "maribelpreite"
DAGSHUB_REPO = "taed2-GAN-Busters"

DEFAULT_MLFLOW_EXPERIMENT = "GAN-Busters Model Selection"

FINAL_MLFLOW_EXPERIMENT = "GAN-Busters Final Training"
FINAL_MLFLOW_RUN_NAME = "final-model-training"

FINAL_EVALUATION_MLFLOW_EXPERIMENT = "GAN-Busters Final Evaluation"
FINAL_EVALUATION_MLFLOW_RUN_NAME = "final-model-evaluation"

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

LABEL_MAPPING = {
    "REAL": 0,
    "FAKE": 1,
}

# Input
INPUT_SHAPE = (32, 32, 3)
NORMALIZE_PIXELS = True

# Base architecture
DEFAULT_CONV_FILTERS = (32, 32)
DEFAULT_DENSE_UNITS = (64,)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_PADDING = "valid"
DEFAULT_CONV_STRIDES = (1, 1)
DEFAULT_POOLING = "max"
DEFAULT_POOL_SIZE = (2, 2)

# Default training configuration
DEFAULT_LOSS = "binary_crossentropy"
DEFAULT_OPTIMIZER = "adam"
DEFAULT_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT = 0.0
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2

# Phase 1 search space
PHASE1_KERNEL_SIZES = (3, 5)
PHASE1_PADDING = ("valid", "same")
PHASE1_POOLING = ("max", "avg")

# Phase 2 search space
PHASE2_LEARNING_RATES = (1e-2, 1e-3, 1e-4)
PHASE2_BATCH_SIZES = (32, 64, 128)
PHASE2_DROPOUT = (0.0, 0.125, 0.25)

# Eval parameters
DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_THRESHOLD = 0.5

# Prediction
PREDICTIONS_DIR = PROJ_ROOT / "data" / "predictions"
DEFAULT_PREDICTIONS_FILE = "predictions.csv"

# Codecarbon
CODECARBON_EXPERIMENT_PROJECT = "CIFAKE_Experiment"
CODECARBON_FINAL_TRAINING_PROJECT = "Final_Model_Training"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

except ModuleNotFoundError:
    pass
