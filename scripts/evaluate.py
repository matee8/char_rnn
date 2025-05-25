#!/usr/bin/env python3

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

from char_rnn import preprocessing, utils
from char_rnn.layers import Dense, Embedding, GRU
from char_rnn.models import Model
from char_rnn.preprocessing import TextVectorizer

logging.basicConfig(level=logging.INFO,
                    format=("%(asctime)s - %(name)s - [%(levelname)s] - "
                            "%(message)s"))

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = (Path(__file__).parent.parent / "models" /
                        "char_rnn_shakespeare.npz")
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "input.txt"


def main(args: argparse.Namespace):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info("Using random seed: %d.", args.seed)

    try:
        logger.info("Loading vocabulary data from '%s'...",
                    args.vocab_data_path)
        full_text_data = utils.load_data_from_file(args.vocab_data_path)
        vectorizer = TextVectorizer()
        vectorizer.fit(full_text_data)
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logger.error("Failed to load or fit TextVectorizer from '%s': %s.",
                     args.vocab_data_path,
                     e,
                     exc_info=True)
        sys.exit(1)

    try:
        encoded_text = vectorizer.encode([full_text_data])[0]
    except ValueError as e:
        logger.error("Error encoding text: %s.", e, exc_info=True)
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
        _, X_test, _, y_test = preprocessing.train_test_split(
            X_batches, y_batches, args.test_size)
    except (ValueError, RuntimeError) as e:
        logger.error("Error splitting dataset: %s.", e, exc_info=True)
        sys.exit(1)

    try:
        model = Model([
            Embedding(V=vectorizer.vocabulary_size, D_e=args.embedding_dim),
            GRU(D_in=args.embedding_dim, D_h=args.hidden_dim),
            Dense(D_in=args.hidden_dim, D_out=vectorizer.vocabulary_size)
        ])
        utils.load_model_weights(model, args.weights_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Failed to initialize model or load weights: %s.",
                     e,
                     exc_info=True)
        sys.exit(1)

    logger.info("Starting model evaluation on the test set.")
    try:
        accuracy = model.evaluate(X_test, y_test)
        logger.info("Model evaluation complete, test accuracy: %.4f", accuracy)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An error occured during model evaluation: %s.",
                     e,
                     exc_info=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Run inference on a pre-trained CharRNN model for text generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--weights-path",
                        type=Path,
                        default=DEFAULT_WEIGHTS_PATH,
                        help="Path to the model weights file (.npz).")
    parser.add_argument(
        "--vocab-data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the original text data file used to build the vocabulary."
    )
    parser.add_argument("--embedding-dim",
                        type=int,
                        default=128,
                        help="Embedding dimension.")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=10,
                        help="Hidden layer dimension.")
    parser.add_argument("--window-size",
                        type=int,
                        default=100,
                        help="Length of the windows passed to the RNN.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of sequences per batch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help=("Proportion of the dataset to include in "
                              "the testing split."))
    args = parser.parse_args()

    if not args.weights_path.is_file():
        raise argparse.ArgumentError(args.weights_path, "file not found.")

    if not args.vocab_data_path.is_file():
        raise argparse.ArgumentError(args.weights_path, "file not found.")

    if args.window_size < 2:
        raise argparse.ArgumentError(args.window_size,
                                     "must be greater than 2.")

    if args.batch_size <= 0:
        raise argparse.ArgumentError(args.batch_size, "must be positive.")

    if not 0.0 < args.test_size < 1.0:
        raise argparse.ArgumentError(args.test_size,
                                     "must be between 0.0 and 1.0")

    return args


if __name__ == "__main__":
    try:
        parsed_args = parse_arguments()
    except argparse.ArgumentError as e:
        logger.error("Argument Error: %s.", e, exc_info=True)
        sys.exit(1)

    main(parsed_args)
