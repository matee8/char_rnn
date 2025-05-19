import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

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


def create_sliding_windows(
        sequence: np.ndarray,
        window_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if sequence.ndim != 1:
        raise ValueError(
            "Input data sequence must be a 1D NumPy array, got "
            f"{sequence.ndim}D array with shape {sequence.shape}.")

    if window_size < 2:
        raise ValueError("Window size must be at least 2.")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    num_total_chars = sequence.shape[0]
    if num_total_chars < window_size:
        raise ValueError("Data is too short for the given window size.")

    T_seq = window_size - 1
    num_windows = num_total_chars - window_size + 1

    X_all = np.zeros((num_windows, T_seq), dtype=sequence.dtype)
    y_all = np.zeros(num_windows, dtype=sequence.dtype)

    for i in range(num_windows):
        window = sequence[i:i + window_size]
        X_all[i] = window[:-1]
        y_all[i] = window[-1]

    logger.debug(
        "Created %d sliding windows. X_all shape: %s, y_all shape: %s.",
        num_windows, X_all.shape, y_all.shape)

    if shuffle:
        rng = np.random.default_rng(seed)

        permutation = rng.permutation(num_windows)
        X_all = X_all[permutation]
        y_all = y_all[permutation]

        logger.debug("Shuffled the windows. Seed: %s.",
                     seed if seed is not None else "None")

    if drop_last:
        num_total_batches = num_windows // batch_size
    else:
        num_total_batches = (num_windows + batch_size - 1) // batch_size

    if num_total_batches == 0:
        raise RuntimeError("No batches to generate.")

    logger.debug(
        "Generating %d mini-batches. Target batch_size=%d, "
        "drop_last=%s. Total windows=%d.", num_total_batches, batch_size,
        drop_last, num_windows)

    for i in range(num_total_batches):
        start = i * batch_size
        end = start + batch_size

        X_batch = X_all[start:end]
        y_batch = y_all[start:end]

        yield X_batch, y_batch
