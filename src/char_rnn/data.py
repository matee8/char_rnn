import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


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


def load_data_from_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found at '{path}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()

        logger.info(
            "Successfully loaded data from '%s'. Total characters: %d.", path,
            len(text_data))

        return text_data
    except IOError as e:
        raise IOError(f"Could not read file at '{path}'.") from e
