from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from gan_busters import config
from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.evaluate import evaluate_model
from gan_busters.modeling.preprocessing import build_dataset
from gan_busters.modeling.tracking import init_tracking
from gan_busters.modeling.train import train_model

'''
The orchestrator divides the train set in 80/20 for model selection
and logs the parameters and metrics in DagsHub/MLflow. 
'''

def run_experiment(args):
    """
    Workflow for hyperparameter exploration and model selection.
    Splits train records into train/val subsets and logs to DagsHub.
    """
    init_tracking()

    accepted_records_path = config.PROCESSED_DATA_DIR / "accepted_records.csv"
    df = pd.read_csv(accepted_records_path)
    train_records = df[df["split"] == "train"]

    # Stratified 80/20 split 
    train_sub, val_sub = train_test_split(
        train_records,
        test_size=0.2,
        stratify=train_records["label"],
        random_state=config.RANDOM_SEED
    )

    val_dataset = build_dataset(val_sub, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        kernel_size=args.kernel_size,
        padding=args.padding,
        pooling=args.pooling,
        learning_rate=args.learning_rate,
        dropout=args.dropout
    )

    run_name = f"exp_k{args.kernel_size}_{args.padding}_{args.pooling}_lr{args.learning_rate}_bs{args.batch_size}"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "kernel_size": args.kernel_size,
            "padding": args.padding,
            "pooling": args.pooling,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "random_seed": config.RANDOM_SEED
        }
        mlflow.log_params(params)

        tracker = EmissionsTracker(project_name="CIFAKE_Experiment", save_to_file=False)
        tracker.start()

        logger.info(f"Starting run: {run_name}")
        train_model(
            model,
            train_sub,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_dataset=val_dataset
        )

        emissions = tracker.stop()
        mlflow.log_metric("emissions_kg_co2", emissions)

        val_metrics = evaluate_model(model, val_sub, batch_size=args.batch_size)
        mlflow.log_metrics(val_metrics)
        logger.success(f"Run completed. Validation Macro F1: {val_metrics['macro_f1']:.4f} | Run ID: {run.info.run_id}")