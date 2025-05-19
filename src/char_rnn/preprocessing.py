import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class TextVectorizer:

    def __init__(self) -> None:
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self._vocab: List[str] = []

    @property
    def vocabulary(self) -> List[str]:
        return self._vocab.copy()

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    def fit(self, text_corpus: str) -> None:
        if not text_corpus:
            raise ValueError("Input text corpus cannot be empty.")

        self._vocab = sorted(list(set(text_corpus)))
        self._char_to_id = {char: i for i, char in enumerate(self._vocab)}
        self._id_to_char = {i: char for i, char in enumerate(self._vocab)}

        logger.info("TextVectorizer fitted with vocabulary size: %d",
                    len(self._vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._vocab:
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
        if not self._vocab:
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
