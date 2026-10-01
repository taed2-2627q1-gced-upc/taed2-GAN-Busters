import kagglehub
from loguru import logger

from gan_busters.config import RAW_DATA_DIR

DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3"


def main():
    output_dir = RAW_DATA_DIR / "cifake"

    logger.info(f"Downloading '{DATASET_SLUG}' from Kaggle...")

    dataset_path = kagglehub.dataset_download(
        DATASET_SLUG,
        output_dir=str(output_dir),
    )

    logger.success(f"Dataset downloaded to {dataset_path}")


if __name__ == "__main__":
    main()
