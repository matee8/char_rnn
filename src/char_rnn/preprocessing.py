import logging
from typing import Dict, List, Optional

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


class Embedding:

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gradients: Optional[np.ndarray] = None
        self._weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self._last_inputs: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_inputs = x

        try:
            return self._weights[x]
        except IndexError as e:
            logger.error("Index %d out of bounds for vocab size %d.",
                         e.args[0], self.vocab_size)
            raise ValueError("Index out of bounds.") from e

    def backward(self, gradient_outputs: np.ndarray) -> None:
        if self._last_inputs is None:
            logger.error("Backward passed called before forward pass.")
            raise RuntimeError("Forward pass must be called before backward "
                               "pass.")

        if gradient_outputs.ndim != 3:
            logger.error(
                "gradient_outputs should be 3D array, got %dD "
                "instead.", gradient_outputs.ndim)
            raise ValueError("gradient_outputs should be a 3D array.")

        if (gradient_outputs.shape[0] != self._last_inputs.shape[0]
                or gradient_outputs.shape[1] != self._last_inputs.shape[1]
                or gradient_outputs.shape[2] != self.embed_dim):
            logger.error("gradient_outputs.shape does not match required "
                         "(batch_size, seq_len, embed_dim) shape.")
            raise ValueError("gradient_outputs.shape does not match output "
                             "shape.")

        if self.gradients is None:
            self.gradients = np.zeros_like(self._weights)

        for i in range(gradient_outputs.shape[0]):
            for j in range(gradient_outputs.shape[1]):
                idx = self._last_inputs[i, j]
                self.gradients[idx] += gradient_outputs[i, j]
