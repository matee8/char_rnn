import argparse
import logging
import sys
from pathlib import Path

from char_rnn import utils
from char_rnn.models import CharRNN
from char_rnn.preprocessing import TextVectorizer

logging.basicConfig(level=logging.INFO,
                    format=("%(asctime)s - %(name)s - [%(levelname)s] - "
                            "%(message)s"))

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path(
    __file__).parent.parent / "models" / "char_rnn_shakespeare.npz"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "input.txt"


def main(args: argparse.Namespace):
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

    missing_chars = {
        char
        for char in args.start_string if char not in vectorizer.vocab
    }
    if missing_chars:
        logger.error("The provided start string contains characters not found "
                     "in the vocabulary.")
        sys.exit(1)

    try:
        model = CharRNN(V=vectorizer.vocabulary_size,
                        D_e=args.embedding_dim,
                        D_h=args.hidden_dim)
        utils.load_model_weights(model, args.weights_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.critical("Failed to initialize model or load weights: %s.",
                        e,
                        exc_info=True)
        sys.exit(1)

    try:
        logger.info(
            "Starting text generation: Generating %d characterrs, "
            "beginning with '%s' (temperature=%.2f)...", args.num_to_generate,
            args.start_string, args.temperature)
        generated_text = model.generate_sequence(vectorizer=vectorizer,
                                                 text=args.start_string,
                                                 n_chars=args.num_to_generate,
                                                 temperature=args.temperature)
        logger.info("Text generation completed successfully.")

        print(f"GENERATED TEXT: {generated_text}")
    except (ValueError, RuntimeError) as e:
        logger.critical("An error occured during text generation: %s",
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
    parser.add_argument("--num-to-generate",
                        type=int,
                        default=64,
                        help="Number of additional characters to generate.")
    parser.add_argument("--embedding-dim",
                        type=int,
                        default=128,
                        help="Embedding dimension.")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=10,
                        help="Hidden layer dimension.")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Temperature for sampling.")
    parser.add_argument("start_string",
                        type=str,
                        help="Initial string to start the text generation.")

    args = parser.parse_args()

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
