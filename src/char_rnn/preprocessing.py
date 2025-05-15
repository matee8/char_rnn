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
            logger.error("Cannot fit TextVectorizer on empty text.")
            raise ValueError("Input text cannot be empty.")

        self.vocab = sorted(list(set(text)))
        self._char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

        logger.info("TextVectorizer fitted with vocabulary size: %d",
                    len(self.vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            logger.error("TextVectorizer encode called before fitting. Call "
                         "fit() first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        if not texts:
            logger.error("No texts provided to encode.")
            raise ValueError("Input texts list is empty.")

        batch_size = len(texts)
        seq_len = len(texts[0])
        sequences = np.zeros((batch_size, seq_len), dtype=int)
        for text_idx, text in enumerate(texts):
            if len(text) != seq_len:
                logger.error("Input texts have differing lenghts.")
                raise ValueError("All texts must have the same length.")

            for char_idx, char in enumerate(text):
                if char not in self._char_to_id:
                    logger.error("Unknown character '%s' in text at index %d.",
                                 char, text_idx)
                    raise ValueError(f"Character '{char}' not in vocabulary.")

                sequences[text_idx, char_idx] = self._char_to_id[char]

        return sequences

    def decode(self, sequences: np.ndarray) -> List[str]:
        if not self.vocab:
            logger.error("TextVectorizer decode called before fitting. Call "
                         "fit() first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        if sequences.ndim != 2:
            logger.error("Input sequences must be 2D, got %s.",
                         sequences.shape)
            raise ValueError("Input array must be 2D.")

        if sequences.dtype != int:
            logger.error("Input sequences must be of type int, got %s.",
                         sequences.dtype)
            raise ValueError("Input sequences must be of type int.")

        texts: List[str] = []

        for seq_idx, sequence in enumerate(sequences):
            chars: List[str] = []

            for idx in sequence:
                idx = int(idx)
                if idx not in self._id_to_char:
                    logger.error("Unknown ID %d in sequence at index %d.", idx,
                                 seq_idx)
                    raise ValueError(f"ID {idx} not in vocabulary.")

                chars.append(self._id_to_char[idx])
            texts.append("".join(chars))

        return texts
