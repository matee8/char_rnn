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


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples mismatch between inputs "
                         f"({X.shape[0]}) and outputs ({y.shape[0]})")

    if X.ndim < 2:
        raise ValueError("Inputs must be at least 2D NumPy array, got "
                         f"{X.ndim}D array with shape {X.shape}.")

    if y.ndim != 2:
        raise ValueError(f"Outputs must be a 2D NumPy array, got {y.ndim}D "
                         f"array with shape {y.shape}")

    if not 0.0 < test_size < 1.0:
        raise ValueError("Test size must be a float between 0.0 and 1.0.")

    num_samples = X.shape[0]
    if num_samples == 0:
        raise ValueError("No samples to split.")

    if num_samples < 2:
        raise ValueError(
            "Not enough samples to create non-empty train and test"
            "sets. Need at least 2 samples.")

    num_test_samples = int(num_samples * test_size)
    num_train_samples = num_samples - num_test_samples

    if num_test_samples == 0:
        num_test_samples = 1
        num_train_samples = num_samples - 1
        logger.warning("Test size resulted in 0 test samples. Setting "
                       "test size to 1 sample.")

    if num_train_samples == 0:
        raise ValueError("Train set will be empty with given test size.")

    indices = np.arange(num_samples)

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    logger.info(
        "Dataset split into train and test, train_size=%d, test_size=%d",
        X_train.shape[0], X_test.shape[0])

    return X_train, X_test, y_train, y_test


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
        X: np.ndarray,
        y: np.ndarray,
        N: int,
        drop_last: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"Input must be a 2D NumPy array, got {X.ndim}D "
                         f"array with shape {X.shape}.")

    if y.ndim != 1:
        raise ValueError(f"Input must be a 1D NumPy array, got {y.ndim}D "
                         f"array with shape {y.shape}.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples mismatch between inputs "
                         f"({X.shape[0]}) and labels ({y.shape[0]})")

    if X.shape[0] == 0:
        raise ValueError("No samples provided to create batches from.")

    if N <= 0:
        raise ValueError("Batch size must be positive.")

    num_samples = X.shape[0]

    if drop_last:
        num_total_batches = num_samples // N
    else:
        num_total_batches = (num_samples + N - 1) // N

    if num_total_batches == 0:
        raise ValueError("No batches can be formed with the current batch size"
                         f"{N} and {num_samples} number of samples.")

    X_batched = np.zeros((num_total_batches, N, X.shape[1]), dtype=X.dtype)
    y_batched = np.zeros((num_total_batches, N), dtype=y.dtype)

    for i in range(num_total_batches):
        start = i * N
        end = start + N

        X_batch = X[start:end]
        y_batch = y[start:end]

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


def shuffle_batches(
        X_batched: np.ndarray,
        y_batched: np.ndarray,
        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if X_batched.ndim != 3:
        raise ValueError(f"Input must be a 3D NumPy array, got "
                         f"{X_batched.ndim}D array with shape "
                         f"{X_batched.shape}.")

    if y_batched.ndim != 2:
        raise ValueError("Output must be a 2D NumPy array, got "
                         f"{y_batched.ndim}D array with shape "
                         f"{y_batched.shape}.")

    if X_batched.shape[0] != y_batched.shape[0]:
        raise ValueError("Number of batches mismatch between inputs ("
                         f"{X_batched.shape[0]}) and outputs ("
                         f"{y_batched.shape[0]}).")

    if X_batched.shape[0] == 0:
        raise ValueError("No batches to shuffle.")

    num_batches = X_batched.shape[0]

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_batches)

    X_batched = X_batched[permutation]
    y_batched = y_batched[permutation]

    logger.info("Shuffled %d batches. Seed: %s.", num_batches,
                seed if seed is not None else "None")

    return X_batched, y_batched
