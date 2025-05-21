import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TextVectorizer:

    def __init__(self) -> None:
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocab)

    def fit(self, text_corpus: str) -> None:
        if not text_corpus:
            raise ValueError("Input text corpus cannot be empty.")

        self.vocab = sorted(list(set(text_corpus)))
        self._char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self._id_to_char = {i: char for i, char in enumerate(self.vocab)}

        logger.info("TextVectorizer fitted with vocabulary size: %d",
                    len(self.vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("Must call fit() before encode().")

        if not texts:
            raise ValueError("Input texts list cannot be empty.")

        N = len(texts)
        T_seq = len(texts[0])
        if T_seq == 0:
            raise ValueError("Texts cannot be empty strings.")

        sequences = np.zeros((N, T_seq), dtype=int)

        for i, text in enumerate(texts):
            if len(text) != T_seq:
                raise ValueError("All texts must have the same length.")

            for j, char in enumerate(text):
                try:
                    sequences[i, j] = self._char_to_id[char]
                except KeyError as e:
                    raise ValueError(
                        f"Character {char} not in vocabulary.") from e

        logger.debug("Encoded texts into ID sequences with shape: %s",
                     sequences.shape)

        return sequences

    def decode(self, sequences: np.ndarray) -> List[str]:
        if not self.vocab:
            raise RuntimeError("Must call fit() before decode()")

        if sequences.ndim != 2:
            raise ValueError(
                "Input sequences shape mismatch. Expected 2D NumPy array"
                f", got shape {sequences.shape}.")

        if not np.issubdtype(sequences.dtype, np.integer):
            raise ValueError(
                "Input sequences dtype mismatch. Expected int, got"
                f"{sequences.dtype}.")

        texts: List[str] = []

        for seq in sequences:
            chars: List[str] = []

            for idx in seq:
                idx = int(idx)
                try:
                    chars.append(self._id_to_char[idx])
                except KeyError as e:
                    raise ValueError(f"ID {idx} not in vocabulary.") from e

            texts.append("".join(chars))

        logger.debug("Decoded ID sequences into %d texts.", len(texts))

        return texts


def create_sliding_windows(
        s: np.ndarray,
        L_w: int,
        shuffle: bool = False,
        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if s.ndim != 1:
        raise ValueError("Input data sequence must be a 1D NumPy array, got "
                         f"{s.ndim}D array with shape {s.shape}")

    if L_w < 2:
        raise ValueError("Window size must be at least 2.")

    num_total_chars = s.shape[0]
    if num_total_chars < L_w:
        raise ValueError("Data is too short for the given window size.")

    L_x = L_w - 1
    num_windows = num_total_chars - L_w + 1

    X = np.zeros((num_windows, L_x), dtype=s.dtype)
    y = np.zeros(num_windows, dtype=s.dtype)

    for i in range(num_windows):
        window = s[i:i + L_w]
        X[i] = window[:-1]
        y[i] = window[-1]

    logger.info("Created %d sliding windows. X shape: %s, y shape: %s.",
                num_windows, X.shape, y.shape)

    if shuffle:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(num_windows)
        X = X[permutation]
        y = y[permutation]
        logger.debug("Shufled the windows. Seed: %d.", seed)

    return X, y


def create_mini_batches(
        X_all: np.ndarray,
        y_all: np.ndarray,
        N: int,
        drop_last: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if X_all.ndim != 2:
        raise ValueError(f"Input must be a 2D NumPy array, got {X_all.ndim}D "
                         f"array with shape {X_all.shape}.")

    if y_all.ndim != 1:
        raise ValueError(f"Input must be a 1D NumPy array, got {y_all.ndim}D "
                         f"array with shape {y_all.shape}.")

    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"Number of instances mismatch between inputs "
                         f"({X_all.shape[0]}) and labels ({y_all.shape[0]})")

    if X_all.shape[0] == 0:
        raise ValueError("No instances provided to create batches from.")

    if N <= 0:
        raise ValueError("Batch size must be positive.")

    num_instances = X_all.shape[0]

    if drop_last:
        num_total_batches = num_instances // N
    else:
        num_total_batches = (num_instances + N - 1) // N

    if num_total_batches == 0:
        raise ValueError("No batches can be formed with the current batch size"
                         f"{N} and {num_instances} number of instances.")

    X_batched = np.zeros((num_total_batches, N, X_all.shape[1]),
                         dtype=X_all.dtype)
    y_batched = np.zeros((num_total_batches, N), dtype=y_all.dtype)

    for i in range(num_total_batches):
        start = i * N
        end = start + N

        X_batch = X_all[start:end]
        y_batch = y_all[start:end]

        if X_batch.shape[0] < N and not drop_last:
            padding_needed = N - X_batch.shape[0]
            X_batch = np.pad(X_batch, ((0, padding_needed), (0, 0)),
                             mode="constant",
                             constant_values=0)
            y_batch = np.pad(y_batch, ((0, padding_needed)),
                             mode="constant",
                             constant_values=0)

            logger.debug("Padded last batch to size %d.", N)

        if X_batch.shape[0] == N:
            X_batched[i] = X_batch
            y_batched[i] = y_batch

    logger.info(
        "Created %d mini-batches, X_batched_shape=%s, "
        "y_batched_shape=%s.", num_total_batches, X_batched.shape,
        y_batched.shape)

    return X_batched, y_batched
