from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
import pandas as pd
import tensorflow as tf

from gan_busters import config
from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.preprocessing import build_dataset


def train_model(model: tf.keras.Model, train_records: pd.DataFrame, epochs: int, batch_size: int, validation_dataset=None):
    """
    Reusable core training loop.
    """
    train_dataset = build_dataset(train_records, batch_size=batch_size, shuffle=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        verbose=1
    )
    return history


def train_final_model(run_id: str):
    """
    Retrieves run params from MLflow, builds a fresh model,
    and trains from scratch on ALL available training data.
    """
    logger.info(f"Hyperparameters from MLflow run ID: {run_id}...")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    model = build_model(
        kernel_size=int(params.get("kernel_size", config.DEFAULT_KERNEL_SIZE)),
        padding=params.get("padding", config.DEFAULT_PADDING),
        pooling=params.get("pooling", config.DEFAULT_POOLING),
        learning_rate=float(params.get("learning_rate", config.DEFAULT_LEARNING_RATE)),
        dropout=float(params.get("dropout", config.DEFAULT_DROPOUT))
    )

    accepted_records_path = config.PROCESSED_DATA_DIR / "accepted_records.csv"
    df = pd.read_csv(accepted_records_path)
    full_train_records = df[df["split"] == "train"]
    epochs = int(params.get("epochs", config.DEFAULT_EPOCHS))
    batch_size = int(params.get("batch_size", config.DEFAULT_BATCH_SIZE))

    logger.info(f"Training final model on {len(full_train_records)} total training images...")
    tracker = EmissionsTracker(project_name="Final_Model_Training", save_to_file=False)
    tracker.start()

    train_model(model, full_train_records, epochs=epochs, batch_size=batch_size)

    emissions = tracker.stop()
    logger.info(f"Emissions recorded: {emissions} kg CO2")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MODELS_DIR / "gan_busters.keras"
    model.save(out_path)
    logger.success(f"Final model saved to {out_path}")