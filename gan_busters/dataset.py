import shutil
from pathlib import Path

import kagglehub
from loguru import logger
from tqdm import tqdm
import typer

from gan_busters.config import RAW_DATA_DIR

app = typer.Typer()

DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_dir: Path = RAW_DATA_DIR / "cifake",
    # ----------------------------------------------
):
    logger.info(f"Downloading '{DATASET_SLUG}' from Kaggle...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    logger.info(f"Downloaded to cache: {cache_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        logger.info(f"{output_dir} already exists, skipping copy.")
    else:
        files = [f for f in cache_path.rglob("*") if f.is_file()]
        for f in tqdm(files, total=len(files), desc="Copying files"):
            dest = output_dir / f.relative_to(cache_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    n_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    logger.success(f"Processing dataset complete. {n_files} files available at {output_dir}")


if __name__ == "__main__":
    app()
