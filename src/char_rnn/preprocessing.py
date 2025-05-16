import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class TextVectorizer:

    def __init__(self) -> None:
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []

    def fit(self, text: str) -> None:
        if not text:
            raise ValueError("Input text cannot be empty.")

        self.vocab = sorted(list(set(text)))
        self._char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

        logger.info("TextVectorizer fitted with vocabulary size: %d",
                    len(self.vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError(
                "Preprocessor must be fit before calling encode().")

        if not texts:
            raise ValueError("Input texts list is empty.")

        batch_size = len(texts)
        seq_len = len(texts[0])
        sequences = np.zeros((batch_size, seq_len), dtype=int)
        for text_idx, text in enumerate(texts):
            if len(text) != seq_len:
                raise ValueError("All texts must have the same length.")

            for char_idx, char in enumerate(text):
                if char not in self._char_to_id:
                    raise ValueError(f"Character '{char}' not in vocabulary.")

                sequences[text_idx, char_idx] = self._char_to_id[char]

        return sequences

    def decode(self, sequences: np.ndarray) -> List[str]:
        if not self.vocab:
            raise RuntimeError(
                "Preprocessor must be fit before calling encode().")

        if sequences.ndim != 2:
            raise ValueError(
                "Input sequences shape mismatch. Expected 2D array"
                f", got {sequences.shape}.")

        if sequences.dtype != int:
            raise ValueError(
                "Input sequences dtype mismatch. Expected int, got"
                f"{sequences.dtype} instead.")

        texts: List[str] = []

        for sequence in sequences:
            chars: List[str] = []

            for idx in sequence:
                idx = int(idx)
                if idx not in self._id_to_char:
                    raise ValueError(f"ID {idx} not in vocabulary.")

                chars.append(self._id_to_char[idx])
            texts.append("".join(chars))

        return texts
