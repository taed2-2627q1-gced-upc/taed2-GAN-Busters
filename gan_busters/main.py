import argparse

from loguru import logger
import tensorflow as tf

from gan_busters import config
from gan_busters.modeling.evaluate import evaluate_final_model
from gan_busters.modeling.experiment import run_experiment
from gan_busters.modeling.predict import predict_images
from gan_busters.modeling.train import train_final_model


def main():
    parser = argparse.ArgumentParser(description="GAN-Busters Central CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: experiment
    exp_parser = subparsers.add_parser("experiment", help="Run model exploration experiment")
    exp_parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    exp_parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    exp_parser.add_argument("--learning-rate", type=float, default=config.DEFAULT_LEARNING_RATE)
    exp_parser.add_argument("--kernel-size", type=int, default=config.DEFAULT_KERNEL_SIZE)
    exp_parser.add_argument("--padding", type=str, default=config.DEFAULT_PADDING)
    exp_parser.add_argument("--pooling", type=str, default=config.DEFAULT_POOLING)
    exp_parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train final model on complete train set using MLflow run ID")
    train_parser.add_argument("--run-id", type=str, required=True, help="Selected MLflow run ID")

    # Subcommand: evaluate
    subparsers.add_parser("evaluate", help="Evaluate final model on test split")

    # Subcommand: predict
    pred_parser = subparsers.add_parser("predict", help="Predict on a single image")
    pred_parser.add_argument("--image-path", type=str, required=True, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment(args)
    elif args.command == "train":
        train_final_model(args.run_id)
    elif args.command == "evaluate":
        evaluate_final_model()
    elif args.command == "predict":
        img_bytes = tf.io.read_file(args.image_path)
        img = tf.io.decode_image(img_bytes, expand_animations=False)
        preds = predict_images(img)
        logger.info(f"Prediction for {args.image_path}: {preds}")


if __name__ == "__main__":
    main()