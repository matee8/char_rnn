import logging
from typing import Dict, List

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

        self.vocab = sorted(set(text))
        self._char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self._id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, texts: List[str]) -> List[List[int]]:
        if not self.vocab:
            logger.error("TextVectorizer not fitted. Call fit first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        try:
            sequences = []
            for text in texts:
                sequences.append([self._char_to_id[ch] for ch in text])
            return sequences
        except KeyError as e:
            logger.error("Unknown character: %s.", e)
            raise ValueError(f"Character {e} not in vocabulary.") from e

    def decode(self, sequences: List[List[int]]) -> List[str]:
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
