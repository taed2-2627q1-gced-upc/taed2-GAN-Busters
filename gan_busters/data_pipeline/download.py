"""Download the raw dataset from Kaggle."""

import kagglehub
from loguru import logger

from gan_busters.config import DATASET_SLUG, RAW_DATA_DIR

def main():
    """Download the dataset from Kaggle into the raw data directory."""
    output_dir = RAW_DATA_DIR

    logger.info(f"Downloading '{DATASET_SLUG}' from Kaggle...")

    dataset_path = kagglehub.dataset_download(
        DATASET_SLUG,
        output_dir=str(output_dir),
    )

    logger.success(f"Dataset downloaded to {dataset_path}")


if __name__ == "__main__":
    main()
