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


def create_batch_sequences(
        s: np.ndarray,
        L_w: int,
        N: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if s.ndim != 1:
        raise ValueError("Input data sequence must be a 1D NumPy array, got "
                         f"{s.ndim}D array with shape {s.shape}.")

    if L_w < 2:
        raise ValueError("Window size must be at least 2.")

    if N <= 0:
        raise ValueError("Batch size must be positive.")

    num_total_chars = s.shape[0]
    if num_total_chars < L_w:
        raise ValueError("Data is too short for the given window size.")

    T_seq = L_w - 1
    num_windows = num_total_chars - L_w + 1

    X_all = np.zeros((num_windows, T_seq), dtype=s.dtype)
    y_all = np.zeros(num_windows, dtype=s.dtype)

    for i in range(num_windows):
        window = s[i:i + L_w]
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
        num_total_batches = num_windows // N
    else:
        num_total_batches = (num_windows + N - 1) // N

    if num_total_batches == 0:
        raise RuntimeError("No batches to generate.")

    logger.debug(
        "Generating %d mini-batches. Target batch_size=%d, "
        "drop_last=%s. Total windows=%d.", num_total_batches, N, drop_last,
        num_windows)

    for i in range(num_total_batches):
        start = i * N
        end = start + N

        X_batch = X_all[start:end]
        y_batch = y_all[start:end]

        yield X_batch, y_batch
