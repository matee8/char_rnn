import logging
from typing import Dict, Iterator, List, Optional, Tuple

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

    L_seq = L_w - 1
    num_windows = num_total_chars - L_w + 1

    X_all = np.zeros((num_windows, L_seq), dtype=s.dtype)
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
