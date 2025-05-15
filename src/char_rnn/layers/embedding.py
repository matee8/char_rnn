import logging
from typing import Optional, Union

import numpy as np

from char_rnn.layers.base import Layer

logger = logging.getLogger(__name__)


class Embedding(Layer):

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        if vocab_size <= 0:
            logger.error("Vocabulary size must be positive, got %d.",
                         vocab_size)
            raise ValueError("Vocabulary size must be positive.")

        if embed_dim <= 0:
            logger.error("Embedding dimension must be positive, got %d.",
                         embed_dim)
            raise ValueError("Embedding dimension must be positive.")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self.gradients = np.zeros_like(self._weights)
        self._last_inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._last_inputs = inputs

        try:
            embedded_outputs = self._weights[inputs]
        except IndexError as e:
            offending_indices = inputs[inputs >= self.vocab_size]
            offending_index: Union[str, int] = (offending_indices[0]
                                                if offending_indices.size > 0
                                                else "unknown")
            logger.error(
                "Index %s out of bounds for vocab_size %d during "
                "embedding.", offending_index, self.vocab_size)
            raise ValueError("Input contains indices out of bounds for "
                             f"vocabulary size {self.vocab_size}. Found index:"
                             f"{offending_index}.") from e

        return embedded_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        if self._last_inputs is None:
            logger.error("Embedding backward pass called before forward pass.")
            raise RuntimeError("Forward pass must be called before backward "
                               "pass for Embedding Layer.")

        if output_gradients.ndim != 3:
            logger.error(
                "Embedding backward pass: output_gradients ndim "
                "mismatch. Expected 3, got %d.", output_gradients.ndim)
            raise ValueError("Output gradients ndim mismatch. Expected 3, got "
                             f"{output_gradients.ndim}.")

        expected_shape = (self._last_inputs.shape[0],
                          self._last_inputs.shape[1], self.embed_dim)
        if output_gradients.shape != expected_shape:
            logger.error(
                "Embedding backward pass: output_gradients shape "
                "mismatch. Expected %s, got %s.", expected_shape,
                output_gradients.shape)
            raise ValueError(
                "Output gradients shape mismatch. Expected "
                f"{expected_shape}, got {output_gradients.shape}.")

        self.gradients.fill(0.0)

        flat_indices = self._last_inputs.ravel()
        reshaped_output_gradients = output_gradients.reshape(
            -1, self.embed_dim)
        np.add.at(self.gradients, flat_indices, reshaped_output_gradients)

        return self.gradients
