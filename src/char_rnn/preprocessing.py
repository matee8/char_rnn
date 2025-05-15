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

        self.vocab = sorted(set(text))
        self._char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            logger.error("TextVectorizer not fitted. Call fit first.")
            raise RuntimeError("Vectorizer has not been fitted on any text.")

        if not texts:
            logger.error("No texts provided to encode.")
            raise ValueError("Input texts list is empty.")

        batch_size = len(texts)
        seq_len = len(texts[0])
        arr = np.zeros((batch_size, seq_len), dtype=int)

        for i, txt in enumerate(texts):
            if len(txt) != seq_len:
                logger.error("Input texts have different lenghts.")
                raise ValueError("All texts must have the same length.")

            for j, ch in enumerate(txt):
                if ch not in self._char_to_id:
                    logger.error("Unknown character: %s", ch)
                    raise ValueError(f"Character '{ch}' not in vocabulary.")

                arr[i, j] = self._char_to_id[ch]

        return arr

    def decode(self, sequences: np.ndarray) -> List[str]:
        if not self.vocab:
            logger.error("Vectorizer not fitted. Call fit() first.")
            raise RuntimeError("Call fit() before decode().")

        if sequences.ndim != 2:
            logger.error("sequences must be 2D, got %s.", sequences.shape)
            raise ValueError("Input array must be 2D.")

        texts: List[str] = []
        for row in sequences:
            chars = []

            for idx in row:
                idx = int(idx)
                if idx not in self._id_to_char:
                    logger.error("Unknown ID: %d.", idx)
                    raise ValueError(f"ID {idx} not in vocabulary.")
                chars.append(self._id_to_char[idx])

            texts.append("".join(chars))

        return texts


class Embedding:

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gradients: Optional[np.ndarray] = None
        self._weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self._last_inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._last_inputs = inputs

        try:
            return self._weights[inputs]
        except IndexError as e:
            logger.error("Index %d out of bounds for vocab size %d.",
                         e.args[0], self.vocab_size)
            raise ValueError("Index out of bounds.") from e

    def backward(self, gradient_outputs: np.ndarray) -> None:
        batch_size, seq_len, _ = gradient_outputs.shape

        if self._last_inputs is None:
            logger.error("Backward passed called before forward pass.")
            raise RuntimeError("Forward pass must be called before backward "
                               "pass.")

        if gradient_outputs.ndim != 3:
            logger.error(
                "gradient_outputs should be 3D array, got %dD "
                "instead.", gradient_outputs.ndim)
            raise ValueError("gradient_outputs should be a 3D array.")

        if (gradient_outputs.shape
                != (self._last_inputs.shape[0], self._last_inputs.shape[1],
                    self.embed_dim)):
            logger.error(
                "Last layer's gradient output shape mismatch. Expected"
                " (%d, %d, %d), got %s.", self._last_inputs.shape[0],
                self._last_inputs.shape[1], self.embed_dim,
                gradient_outputs.shape)
            raise ValueError(
                "Last layer's gradient output shape mismatch. Expected"
                f" ({self._last_inputs.shape[0]}, "
                f"{self._last_inputs.shape[1]}, {self.embed_dim}), "
                f"got {gradient_outputs.shape}.")

        if self.gradients is None:
            self.gradients = np.zeros_like(self._weights)

        for i in range(batch_size):
            for j in range(seq_len):
                idx = self._last_inputs[i, j]
                self.gradients[idx] += gradient_outputs[i, j]
