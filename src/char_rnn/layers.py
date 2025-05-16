import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Layer(ABC):

    @abstractmethod
    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        pass


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

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
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


class Recurrent(Layer):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        if input_dim <= 0:
            logger.error("Input dimension must be positive, got %d.",
                         input_dim)
            raise ValueError("Input dimension must be positive.")

        if hidden_dim <= 0:
            logger.error("Hidden dimension must be positive, got %d.",
                         hidden_dim)
            raise ValueError("Hidden dimension must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._weights_input_hidden = (np.random.randn(input_dim, hidden_dim) *
                                      0.01)
        self._weights_hidden_hidden = (
            np.random.randn(hidden_dim, hidden_dim) * 0.01)
        self._hidden_bias = np.zeros((1, hidden_dim))

        self.gradients_input_hidden = np.zeros_like(self._weights_input_hidden)
        self.gradients_hidden_hidden = (np.zeros_like(
            self._weights_hidden_hidden))
        self.gradients_bias = np.zeros_like(self._hidden_bias)

        self._last_inputs: Optional[np.ndarray] = None
        self._last_hidden_states: Optional[np.ndarray] = None

    def forward_step(self, current_input: np.ndarray,
                     previous_hidden_state: np.ndarray) -> np.ndarray:
        if current_input.ndim == 1:
            current_input = current_input.reshape(1, -1)

        if previous_hidden_state.ndim == 1:
            previous_hidden_state = previous_hidden_state.reshape(1, -1)

        if current_input.shape[1] != self.input_dim:
            logger.error(
                "Recurrent forward step: input shape mismatch. "
                "Expected input dimension %d, got %s instead.", self.input_dim,
                current_input.shape[1])
            raise ValueError(f"Input shape mismatch. Expected input dimension "
                             f"{self.input_dim}, got {current_input.shape} "
                             "instead.")

        if previous_hidden_state.shape[1] != self.hidden_dim:
            logger.error(
                "Recurrent forward step: previous hidden state shape "
                "mismatch. Expected hidden dimension %d, got %s "
                "instead.", self.hidden_dim, previous_hidden_state.shape)
            raise ValueError("Previous hidden state shape mismatch. Expected "
                             f"hidden dimension {self.hidden_dim}, got "
                             f"{previous_hidden_state.shape} instead.")

        if current_input.shape[0] != previous_hidden_state.shape[0]:
            logger.error(
                "Recurrent forward step: batch size mismatch between "
                "current input (%s) and previous hidden state (%s).",
                current_input.shape[0], previous_hidden_state.shape[0])
            raise ValueError("Batch size mismatch. Expected "
                             f"{current_input.shape[0]}, got "
                             f"{previous_hidden_state.shape[0]} instead.")

        hidden_state_unactivated = (
            current_input @ self._weights_input_hidden +
            previous_hidden_state @ self._weights_hidden_hidden +
            self._hidden_bias)
        current_hidden_state = np.tanh(hidden_state_unactivated)

        return current_hidden_state

    def forward(self,
                inputs: np.ndarray,
                initial_state: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        if inputs.ndim != 3:
            logger.error(
                "Recurrent forward pass: input sequence ndim mismatch"
                ". Expected 3D array, got %s instead.", inputs.shape)
            raise ValueError(
                "Input sequence ndim mismatch. Expected 3D array, "
                f"got {inputs.shape} instead.")

        batch_size, seq_len, input_dim = inputs.shape

        if input_dim != self.input_dim:
            logger.error(
                "Recurrent forward pass: input sequence dimension "
                "mismatch. Expected shape[-1] as %d, got %d instead.",
                self.input_dim, input_dim)
            raise ValueError("Input dimension mismatch. Expected shape[-1] to "
                             f"be {self.input_dim}, got {input_dim} instead.")

        if initial_state is None:
            current_hidden_state = (np.zeros((batch_size, self.hidden_dim)))
        else:
            if initial_state.shape != (batch_size, self.hidden_dim):
                logger.error(
                    "Recurrent forward step: initial hidden state "
                    "shape mismatch. Expected shape (%d, %d), got "
                    "%s instead.", batch_size, self.hidden_dim,
                    initial_state.shape)
                raise ValueError(
                    "Initial hidden state shape mismatch. Expected"
                    f" shape ({batch_size}, {self.hidden_dim}), "
                    f"got {initial_state.shape} instead.")
            current_hidden_state = initial_state

        self._last_inputs = inputs
        self._last_hidden_states = (np.zeros(
            (batch_size, seq_len + 1, self.hidden_dim)))
        self._last_hidden_states[:, 0, :] = current_hidden_state

        for t in range(seq_len):
            current_input = inputs[:, t, :]
            current_hidden_state = (self.forward_step(current_input,
                                                      current_hidden_state))
            self._last_hidden_states[:, t + 1, :] = current_hidden_state

        return current_hidden_state

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        if self._last_inputs is None or self._last_hidden_states is None:
            logger.error("Recurrent backward pass called before forward pass "
                         "or history is missing.")
            raise RuntimeError(
                "Forward pass must be called to populate history"
                " before backward pass.")

        if output_gradients.ndim != 2:
            logger.error(
                "Output gradients ndim mismatch in backward pass. "
                "Expected 2D, got %s.", output_gradients.shape)
            raise ValueError("Output gradients must be 2D array.")

        batch_size, seq_len, _ = self._last_inputs.shape
        expected_gradient_shape = (batch_size, self.hidden_dim)
        if output_gradients.shape != expected_gradient_shape:
            msg = ("Final hidden state gradients shape mismatch. "
                   f"Expected {expected_gradient_shape}, got "
                   f"{output_gradients.shape}")
            logger.error(msg)
            raise ValueError(msg)

        self.gradients_input_hidden.fill(0.0)
        self.gradients_hidden_hidden.fill(0.0)
        self.gradients_bias.fill(0.0)

        input_sequence_gradients = np.zeros_like(self._last_inputs)
        current_gradients = output_gradients.copy()

        for t in reversed(range(seq_len)):
            current_hidden_states = self._last_hidden_states[:, t + 1, :]
            previous_hidden_states = self._last_hidden_states[:, t, :]
            current_inputs = self._last_inputs[:, t, :]

            tanh_derivative = 1 - current_hidden_states**2

            gradient_pre_tanh = current_gradients * tanh_derivative

            self.gradients_input_hidden += current_inputs.T @ gradient_pre_tanh
            self.gradients_hidden_hidden += (
                previous_hidden_states.T @ gradient_pre_tanh)
            self.gradients_bias += gradient_pre_tanh.sum(axis=0, keepdims=True)

            input_sequence_gradients[:, t, :] = (
                tanh_derivative @ self._weights_input_hidden.T)
            current_gradients = tanh_derivative @ self._weights_hidden_hidden.T

        return input_sequence_gradients


class Dense(Layer):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if input_dim <= 0:
            logger.error("Input dimension must be positive, got %d.",
                         input_dim)
            raise ValueError("Input dimension must be positive.")

        if output_dim <= 0:
            logger.error("Output dimension must be positive, got %d.",
                         output_dim)
            raise ValueError("Output dimension must be positive.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._weights = np.random.randn(input_dim, output_dim) * 0.01
        self._bias = np.zeros((1, output_dim))

        self.gradients: Optional[np.ndarray] = None
        self.gradients_bias: Optional[np.ndarray] = None

        self._last_inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        if inputs.ndim != 2:
            logger.error(
                "Dense layer forward pass: input data ndim mismatch. "
                "Expected 2, got %d instead.", inputs.ndim)
            raise ValueError("Inputs ndim mismatch. Expected 2, got "
                             f"{inputs.ndim} instead.")

        if inputs.shape[-1] != self.input_dim:
            logger.error(
                "Dense layer forward pass: input data shape[-1] "
                "mismatch. Expected %d, got %s instead.", self.input_dim,
                inputs.shape)
            raise ValueError("Input data shape mismatch. Expected shape[-1] "
                             f"to be {self.input_dim}, got {inputs.shape} "
                             "instead")

        self._last_inputs = inputs

        outputs_before_activation = inputs @ self._weights + self._bias

        stable_outputs = (
            np.exp(outputs_before_activation -
                   np.max(outputs_before_activation, axis=1, keepdims=True)))

        return stable_outputs / np.sum(stable_outputs, axis=1, keepdims=True)
