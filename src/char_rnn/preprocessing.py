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
            logger.error("Cannot fit on empty text.")
            raise ValueError("Input text cannot be empty.")

        self.vocab = sorted(list(set(text)))
        self._char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            logger.error("TextVectorizer not fitted. Call fit first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        sequences = []
        for text in texts:
            try:
                sequences.append([self._char_to_id[ch] for ch in text])
            except KeyError as e:
                logger.error("Unknown character: %s.", e.args[0])
                raise ValueError(
                    f"Character {e.args[0]} not in vocabulary.") from e

        try:
            return np.array(sequences)
        except ValueError as e:
            logger.error("Input texts in batch are not the same length. "
                         "Stopping.")
            raise ValueError("Input texts are not the same length.") from e

    def decode(self, sequences: np.ndarray) -> List[str]:
        if not self.vocab:
            logger.error("TextVectorizer not fitted. Call fit first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        try:
            texts = []
            for seq in sequences:
                texts.append("".join([self._id_to_char[id] for id in seq]))
            return texts
        except KeyError as e:
            logger.error("Unknown character: %s.", e)
            raise ValueError(f"Character {e} not in vocabulary.") from e
