#!/usr/bin/env python3

import argparse
import logging
import random
import time

from pathlib import Path
from typing import List

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
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_MODEL_DIR = "models"
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

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    model_dir = project_root / args.model_dir

    data_file_path = data_dir / args.data_filename

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Could not create directories: %s.", e, exc_info=True)
        return

    if not data_file_path.exists():
        logger.info("Dataset not found. Downloading from %s...", args.data_url)
        try:
            download_file(args.data_url, data_file_path)
        except requests.exceptions.RequestException as e:
            logger.error("Download failed: %s.", e, exc_info=True)
            return
        except IOError as e:
            logger.error("Could not write to file: %s.", e, exc_info=True)
            return
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
        return

    if not text_data:
        logger.error("Loaded data is empty. Exiting.")
        return

    vectorizer = TextVectorizer()
    vectorizer.fit(text_data)

    try:
        encoded_text = vectorizer.encode([text_data])[0]
    except ValueError as e:
        logger.error("Error encoding text: %s.", e, exc_info=True)
        return

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
        return

    logger.info(
        "Starting training, num_epochs=%d, batch_size=%d, "
        "window_size=%d.", args.num_epochs, args.batch_size, args.window_size)

    for epoch in range(1, args.num_epochs + 1):
        epoch_losses: List[float] = []

        try:
            batch_generator = preprocessing.create_batch_sequences(
                s=encoded_text,
                L_w=args.window_size,
                N=args.batch_size,
                shuffle=True,
                seed=args.seed + epoch,
                drop_last=True)
        except (ValueError, RuntimeError) as e:
            logger.error("Error creating batch generator: %s.",
                         e,
                         exc_info=True)
            continue

        for i, (x, y) in enumerate(batch_generator):
            try:
                print(x.shape)
                loss = model.train_step(x, y)
                epoch_losses.append(loss)

                if (i + 1) % args.log_interval == 0:
                    logger.info("Epoch %d/%d - Batch %d - Loss: %.4f", epoch,
                                args.num_epochs, i + 1, loss)
            except Exception as e:
                logger.error("Error during training step: %s.",
                             e,
                             exc_info=True)
                continue

        avg_epoch_loss = np.mean(epoch_losses)
        logger.info("Epoch %d/%d - Loss: %.4f", epoch, args.num_epochs,
                    avg_epoch_loss)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / (args.model_filename + timestamp)

    logger.info("Training completed. Saving model weights to %s...",
                model_path)

    try:
        utils.save_model_weights(model, model_path)
    except Exception as e:
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
                           type=str,
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
                            type=int,
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

    output_args = parser.add_argument_group(
        "Output and logging configuration")
    output_args.add_argument("--model-dir",
                             type=str,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
