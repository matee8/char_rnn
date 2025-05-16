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
            raise ValueError(
                f"Vocabulary size must be positive, got {vocab_size}.")

        if embed_dim <= 0:
            raise ValueError(
                f"Embedding dimension must be positive, got {embed_dim}.")

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
            raise ValueError("Input contains indices out of bounds for "
                             f"vocabulary size {self.vocab_size}. Found index:"
                             f"{offending_index}.") from e

        return embedded_outputs

    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        if self._last_inputs is None:
            raise RuntimeError("Forward pass must be called before backward.")

        if output_gradients.ndim != 3:
            raise ValueError(
                "Output gradients ndim mismatch. Expected 3D array"
                f", got {output_gradients.shape}.")

        expected_shape = (self._last_inputs.shape[0],
                          self._last_inputs.shape[1], self.embed_dim)
        if output_gradients.shape != expected_shape:
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
            raise ValueError(
                f"Input dimension must be positive, got {input_dim}.")

        if hidden_dim <= 0:
            raise ValueError(
                f"Hidden dimension must be positive, got {hidden_dim}.")

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
            raise ValueError(f"Input shape mismatch. Expected input dimension "
                             f"{self.input_dim}, got {current_input.shape}.")

        if previous_hidden_state.shape[1] != self.hidden_dim:
            raise ValueError("Previous hidden state shape mismatch. Expected "
                             f"hidden dimension {self.hidden_dim}, got "
                             f"{previous_hidden_state.shape}.")

        if current_input.shape[0] != previous_hidden_state.shape[0]:
            raise ValueError("Batch size mismatch. Expected "
                             f"{current_input.shape[0]}, got "
                             f"{previous_hidden_state.shape[0]}.")

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
            raise ValueError(
                "Input sequence ndim mismatch. Expected 3D array, "
                f"got {inputs.shape}.")

        batch_size, seq_len, input_dim = inputs.shape

        if input_dim != self.input_dim:
            raise ValueError("Input dimension mismatch. Expected shape[-1] to "
                             f"be {self.input_dim}, got {input_dim}.")

        if initial_state is None:
            current_hidden_state = (np.zeros((batch_size, self.hidden_dim)))
        else:
            if initial_state.shape != (batch_size, self.hidden_dim):
                raise ValueError(
                    "Initial hidden state shape mismatch. Expected"
                    f" shape ({batch_size}, {self.hidden_dim}), "
                    f"got {initial_state.shape}.")
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
            raise RuntimeError(
                "Forward pass must be called to populate history"
                " before backward pass.")

        if output_gradients.ndim != 2:
            raise ValueError(
                "Output gradients ndim mismatch. Expected 2D array"
                f", got {output_gradients.shape} instead.")

        batch_size, seq_len, _ = self._last_inputs.shape
        expected_gradient_shape = (batch_size, self.hidden_dim)
        if output_gradients.shape != expected_gradient_shape:
            raise ValueError("Final hidden state gradients shape mismatch. "
                             f"Expected {expected_gradient_shape}, got "
                             f"{output_gradients.shape}.")

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
            raise ValueError(
                f"Input dimension must be positive, got {input_dim}.")

        if output_dim <= 0:
            raise ValueError(
                f"Output dimension must be positive, got {output_dim}.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._weights = np.random.randn(input_dim, output_dim) * 0.01
        self._bias = np.zeros((1, output_dim))

        self.gradients: Optional[np.ndarray] = None
        self.gradients_bias: Optional[np.ndarray] = None

        self._last_inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        if inputs.ndim != 2:
            raise ValueError("Inputs ndim mismatch. Expected 2D array, got "
                             f"{inputs.shape} instead.")

        if inputs.shape[-1] != self.input_dim:
            raise ValueError("Input data shape mismatch. Expected shape[-1] "
                             f"to be {self.input_dim}, got {inputs.shape}.")

        self._last_inputs = inputs

        outputs_before_activation = inputs @ self._weights + self._bias

        stable_outputs = (
            np.exp(outputs_before_activation -
                   np.max(outputs_before_activation, axis=1, keepdims=True)))

        return stable_outputs / np.sum(stable_outputs, axis=1, keepdims=True)
