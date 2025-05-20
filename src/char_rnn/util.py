import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data_from_file(filepath: str) -> str:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at '{filepath}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()

        logger.info(
            "Successfully loaded data from '%s'. Total characters: %d.",
            filepath, len(text_data))

        return text_data
    except IOError as e:
        raise IOError(f"Could not read file at '{filepath}'.") from e
