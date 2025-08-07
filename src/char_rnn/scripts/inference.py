#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from char_rnn import data, models
from char_rnn.models import Model
from char_rnn.preprocessing import TextVectorizer

logging.basicConfig(level=logging.INFO,
                    format=("%(asctime)s - %(name)s - [%(levelname)s] - "
                            "%(message)s"))

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = (Path(__file__).parent.parent / "models" /
                        "char_rnn_shakespeare.npz")
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "input.txt"


def generate_text(model: Model,
                  vectorizer: TextVectorizer,
                  start_string: str,
                  n_chars: int,
                  temperature: float = 1.0) -> str:
    try:
        encoded_start_string = vectorizer.encode([start_string])[0]
    except ValueError as e:
        raise ValueError("Could not encode start string.") from e

    generated_text = encoded_start_string

    for _ in range(n_chars):
        y_proba = model.predict(generated_text)

        if y_proba is None:
            raise RuntimeError("Model prediction returned None unexpectedly "
                               "during text generation.")

        if temperature == 1.0:
            next_char = np.argmax(y_proba, axis=1)[0]
        else:
            y_proba_clipped = np.clip(y_proba, 1e-9, 1.0)
            log_probas = np.log(y_proba_clipped)
            scaled_log_probas = log_probas / temperature

            exp_scaled_log_probas = (
                np.exp(scaled_log_probas -
                       np.max(scaled_log_probas, axis=1, keepdims=True)))
            final_probas = (
                exp_scaled_log_probas /
                np.sum(exp_scaled_log_probas, axis=1, keepdims=True))

            next_char = np.random.choice(vectorizer.vocabulary_size,
                                         p=final_probas[0])

        generated_text = np.append(generated_text, next_char)

    logger.info("Text generation completed successfully.")

    return vectorizer.decode(generated_text.reshape(1, -1))[0]


def main(args: argparse.Namespace):
    try:
        logger.info("Loading vocabulary data from '%s'...",
                    args.vocab_data_path)
        full_text_data = data.load_data_from_file(args.vocab_data_path)
        vectorizer = TextVectorizer()
        vectorizer.fit(full_text_data)
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logger.error("Failed to load or fit TextVectorizer from '%s': %s.",
                     args.vocab_data_path,
                     e,
                     exc_info=True)
        sys.exit(1)

    missing_chars = {
        char
        for char in args.start_string if char not in vectorizer.vocab
    }
    if missing_chars:
        logger.error("The provided start string contains characters not found "
                     "in the vocabulary.")
        sys.exit(1)

    try:
        model = models.create_char_rnn(V=vectorizer.vocabulary_size,
                                       D_e=args.embedding_dim,
                                       D_h=args.hidden_dim)
        model.load_weights(args.weights_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Failed to initialize model or load weights: %s.",
                     e,
                     exc_info=True)
        sys.exit(1)

    try:
        generated_text = generate_text(model=model,
                                       vectorizer=vectorizer,
                                       start_string=args.start_string,
                                       n_chars=args.num_to_generate,
                                       temperature=args.temperature)
    except (ValueError, RuntimeError) as e:
        logger.error("An error occured during text generation: %s",
                     e,
                     exc_info=True)
        sys.exit(1)

    print(f"GENERATED TEXT: {generated_text}")


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
    parser.add_argument("--num-to-generate",
                        type=int,
                        default=64,
                        help="Number of additional characters to generate.")
    parser.add_argument("--embedding-dim",
                        type=int,
                        default=16,
                        help="Dimension of character embeddings.")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=128,
                        help="Dimension of hidden states.")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Temperature for sampling.")
    parser.add_argument("start_string",
                        type=str,
                        help="Initial string to start the text generation.")

    args = parser.parse_args()

    if not args.weights_path.is_file():
        raise argparse.ArgumentError(args.weights_path, "file not found.")

    if not args.vocab_data_path.is_file():
        raise argparse.ArgumentError(args.weights_path, "file not found.")

    if args.temperature <= 0:
        raise argparse.ArgumentError(args.temperature,
                                     "must be a positive value.")

    if args.num_to_generate <= 0:
        raise argparse.ArgumentError(args.num_to_generate,
                                     "must be a positive value.")

    return args


if __name__ == "__main__":
    try:
        parsed_args = parse_arguments()
    except argparse.ArgumentError as e:
        logger.error("Argument Error: %s.", e, exc_info=True)
        sys.exit(1)

    main(parsed_args)
