#!/usr/bin/env python3

import argparse
import logging
import random
import time
import sys

from pathlib import Path

import numpy as np
import requests

from char_rnn import preprocessing, utils
from char_rnn.losses import SparseCategoricalCrossEntropy
from char_rnn.models import CharRNN
from char_rnn.optimizers import Adam
from char_rnn.preprocessing import TextVectorizer

logging.basicConfig(level=logging.INFO,
                    format=("%(asctime)s - %(name)s - [%(levelname)s] - "
                            "%(message)s"))

logger = logging.getLogger(__name__)

DEFAULT_DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                    "master/data/tinyshakespeare/input.txt")
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_DATA_FILENAME = "input.txt"
DEFAULT_MODEL_FILENAME = "char_rnn_shakespeare"


def download_file(url: str, dest: Path) -> None:
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            with open(temp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        temp_dest.rename(dest)
        logger.info("File downloaded successfully to %s.", dest)
    finally:
        if temp_dest.exists():
            temp_dest.unlink()


def main(args: argparse.Namespace):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info("Using random seed: %d.", args.seed)

    data_file_path = args.data_dir / args.data_filename
    model_file_base_path = args.model_dir / args.model_filename

    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Could not create directories: %s.", e, exc_info=True)
        sys.exit(1)

    if not data_file_path.exists():
        logger.info("Dataset not found. Downloading from %s...", args.data_url)
        try:
            download_file(args.data_url, data_file_path)
        except requests.exceptions.RequestException as e:
            logger.error("Download failed: %s.", e, exc_info=True)
            sys.exit(1)
        except IOError as e:
            logger.error("Could not write to file: %s.", e, exc_info=True)
            sys.exit(1)
    else:
        logger.info("Dataset already exists at %s. Skipping download.",
                    data_file_path)

    try:
        text_data = utils.load_data_from_file(data_file_path)
    except (FileNotFoundError, IOError) as e:
        logger.error("Error loading data file '%s': %s.",
                     data_file_path,
                     e,
                     exc_info=True)
        sys.exit(1)

    if not text_data:
        logger.error("Loaded data is empty. Exiting.")
        sys.exit(1)

    try:
        vectorizer = TextVectorizer()
        vectorizer.fit(text_data)
    except (ValueError, RuntimeError) as e:
        logger.error("Failed to preprocess data: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        encoded_text = vectorizer.encode([text_data])[0]
    except ValueError as e:
        logger.error("Error encoding text: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        model = CharRNN(V=vectorizer.vocabulary_size,
                        D_e=args.embedding_dim,
                        D_h=args.hidden_dim)
        optimizer = Adam(eta=args.learning_rate)
        loss_fn = SparseCategoricalCrossEntropy()
        model.compile(optimizer, loss_fn)
    except ValueError as e:
        logger.error("Error initializing model components: %s.",
                     e,
                     exc_info=True)
        sys.exit(1)

    try:
        X_all, y_all = preprocessing.create_sliding_windows(
            s=encoded_text, L_w=args.window_size)
    except (ValueError, RuntimeError) as e:
        logger.error("Error creating sliding windows: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        X_batches, y_batches = preprocessing.create_mini_batches(
            X=X_all, y=y_all, N=args.batch_size, drop_last=True)
    except (ValueError, RuntimeError) as e:
        logger.error("Error creating batches: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        X_train, _, y_train, _ = preprocessing.train_test_split(
            X_batches, y_batches, 1 - args.train_size)
        X_train, X_valid, y_train, y_valid = (preprocessing.train_test_split(
            X_train, y_train, args.validation_size))
    except (ValueError, RuntimeError) as e:
        logger.error("Error splitting dataset: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        model.fit(X_batches=X_train,
                  y_batches=y_train,
                  num_epochs=args.num_epochs,
                  log_interval=args.log_interval,
                  X_val=X_valid,
                  y_val=y_valid,
                  seed=args.seed)
    except (ValueError, RuntimeError) as e:
        logger.error("Failed to train model: %s.", e, exc_info=True)
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = model_file_base_path.with_suffix(f".{timestamp}.npz")

    logger.info("Training completed. Saving model weights to %s...",
                model_path)

    try:
        utils.save_model_weights(model, model_path)
    except IOError as e:
        logger.error("Failed to save model weights: %s.", e, exc_info=True)
        return


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Char-RNN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_args = parser.add_argument_group("Data configuration")
    data_args.add_argument("--data-url",
                           type=str,
                           default=DEFAULT_DATA_URL,
                           help="URL to download the dataset from.")
    data_args.add_argument("--data-dir",
                           type=Path,
                           default=DEFAULT_DATA_DIR,
                           help="Directory to store downloaded data.")
    data_args.add_argument("--data-filename",
                           type=str,
                           default=DEFAULT_DATA_FILENAME,
                           help="Filename for the dataset within data_dir.")

    model_args = parser.add_argument_group("Model hyperparameters")
    model_args.add_argument("--embedding-dim",
                            type=int,
                            default=128,
                            help="Dimension of character embeddings.")
    model_args.add_argument("--hidden-dim", type=int, default=10, help="")
    model_args.add_argument("--learning-rate",
                            type=float,
                            default=0.001,
                            help="Learning rate for the Adam optimizer.")

    training_args = parser.add_argument_group("Training parameters")
    training_args.add_argument("--window-size",
                               type=int,
                               default=100,
                               help="Length of the windows passed to the RNN.")
    training_args.add_argument("--batch-size",
                               type=int,
                               default=32,
                               help="Number of sequences per batch.")
    training_args.add_argument("--num-epochs",
                               type=int,
                               default=10,
                               help="Number of training epochs.")
    training_args.add_argument("--seed",
                               type=int,
                               default=42,
                               help="Random seed.")
    training_args.add_argument("--train-size",
                               type=float,
                               default=0.8,
                               help=("Proportion of the dataset to include in "
                               "the training split."))
    training_args.add_argument("--validation-size",
                               type=float,
                               default=0.2,
                               help=("Proportion of the training dataset to"
                                     "include in the validation split."))

    output_args = parser.add_argument_group("Output and logging configuration")
    output_args.add_argument("--model-dir",
                             type=Path,
                             default=DEFAULT_MODEL_DIR,
                             help="Directory to save trained models.")
    output_args.add_argument("--model-filename",
                             type=str,
                             default=DEFAULT_MODEL_FILENAME,
                             help="Filename for the model within model_dir.")

    output_args.add_argument("--log-interval",
                             type=int,
                             default=10,
                             help="Log training loss every N batches.")

    args = parser.parse_args()

    if args.embedding_dim <= 0:
        raise argparse.ArgumentError(args.embedding_dim, "must be positive.")

    if args.hidden_dim <= 0:
        raise argparse.ArgumentError(args.hidden_dim, "must be positive.")

    if args.learning_rate <= 0:
        raise argparse.ArgumentError(args.learning_rate, "must be positive.")

    if args.window_size < 2:
        raise argparse.ArgumentError(args.window_size,
                                     "must be greater than 2.")

    if args.batch_size <= 0:
        raise argparse.ArgumentError(args.batch_size, "must be positive.")

    if args.num_epochs <= 0:
        raise argparse.ArgumentError(args.num_epochs, "must be positive.")

    if args.log_interval <= 0:
        raise argparse.ArgumentError(args.log_interval, "must be positive.")

    if not 0.0 < args.train_size < 1.0:
        raise argparse.ArgumentError(args.train_size, "must be between 0.0 and 1.0")

    if not 0.0 < args.validation_size < 1.0:
        raise argparse.ArgumentError(args.validation_size, "must be between 0.0 and 1.0")

    return args


if __name__ == "__main__":
    try:
        parsed_args = parse_arguments()
    except argparse.ArgumentError as e:
        logger.error("Argument error: %s.", e, exc_info=True)
        sys.exit(1)

    main(parsed_args)
