import logging
from pathlib import Path
from typing import Optional, Tuple

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
        shuffle: bool = False,
        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if sequence.ndim != 1:
        raise ValueError(
            "Input data sequence must be a 1D NumPy array, got "
            f"{sequence.ndim}D array with shape {sequence.shape}.")

    if window_size < 2:
        raise ValueError("Window size must be at least 2.")

    num_total_chars = sequence.shape[0]
    if num_total_chars < window_size:
        raise ValueError("Data is too short for the given window size.")

    num_windows = num_total_chars - window_size + 1

    X_len = window_size - 1

    X = np.zeros((num_windows, X_len), dtype=sequence.dtype)
    y = np.zeros(num_windows, dtype=sequence.dtype)

    for i in range(num_windows):
        window = sequence[i:i + window_size]
        X[i] = window[:-1]
        y[i] = window[-1]

    logger.info("Created %d sliding windows. X shape: %s, y shape: %s.",
                num_windows, X.shape, y.shape)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)

        permutation = np.random.permutation(num_windows)
        X = X[permutation]
        y = y[permutation]

        logger.info("Shuffled the windows. Seed: %s.",
                    seed if seed is not None else "None")

    return X, y
