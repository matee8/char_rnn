import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Recurrent:

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            logger.error("Input size or hidden size initialized to "
                         "non-positive.")
            raise ValueError("Input size and hidden size must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._weights_input_hidden = np.random.randn(input_dim,
                                                     hidden_dim) * 0.01
        self._weights_hidden_hidden = np.random.randn(hidden_dim,
                                                      hidden_dim) * 0.01
        self._hidden_bias = np.zeros((1, hidden_dim))

        self.gradients_input_hidden: Optional[np.ndarray] = None
        self.gradients_hidden_hidden: Optional[np.ndarray] = None
        self.gradients_hidden_bias: Optional[np.ndarray] = None

        self._last_inputs: Optional[np.ndarray] = None
        self._last_hidden_states: Optional[np.ndarray] = None

    def forward_step(self,
                     inputs: np.ndarray,
                     previous_hidden: np.ndarray | None = None) -> np.ndarray:
        if inputs is not None and inputs.ndim == 1:
            inputs = inputs.reshape(1, inputs.shape[-1], copy=False)

        if previous_hidden is not None and previous_hidden.ndim == 1:
            previous_hidden = (previous_hidden.reshape(
                1, previous_hidden.shape[-1]))

        if inputs.shape[1] != self.input_dim:
            logger.error(
                "Input shape mismatch. Expected seq_len (shape[1]) to "
                "be %d, got %d instead.", self.input_dim, inputs.shape[1])
            raise ValueError("Input shape mismatch. Expected seq_len "
                             f"(shape[1]) to be {self.input_dim}, got "
                             f"{inputs.shape[1]}")

        if (previous_hidden is not None
                and previous_hidden.shape[-1] != self.hidden_dim):
            logger.error(
                "Previous hidden state shape mismatch. Expected hidden_dim "
                "(shape[1]) to be %d, got %d instead.", self.hidden_dim,
                previous_hidden.shape[-1])
            raise ValueError(
                "Previous hidden state shape mismatch. Expected hidden_dim "
                f"(shape[1])to be {self.hidden_dim}, got"
                f"{previous_hidden.shape[-1]} instead.")

        if previous_hidden is None:
            previous_hidden = (np.zeros(
                self.hidden_dim * inputs.shape[0]).reshape(inputs.shape[0],
                                                           self.hidden_dim,
                                                           copy=False))

        if self._last_inputs is None:
            self._last_inputs = inputs[:, np.newaxis, :]
        else:
            self._last_inputs = (np.append(self._last_inputs,
                                           inputs[:, np.newaxis, :],
                                           axis=1))

        if self._last_hidden_states is None:
            self._last_hidden_states = previous_hidden[:, np.newaxis, :]

        hidden_states = np.tanh(inputs @ self._weights_input_hidden +
                                previous_hidden @ self._weights_hidden_hidden +
                                self._hidden_bias)

        self._last_hidden_states = (np.append(self._last_hidden_states,
                                              hidden_states[:, np.newaxis, :],
                                              axis=1))

        return hidden_states

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = inputs.shape

        self._last_inputs = inputs

        hidden_states = np.zeros((batch_size, self.hidden_dim))
        self._last_hidden_states = np.zeros(
            (batch_size, seq_len + 1, self.hidden_dim))
        self._last_hidden_states[:, 0] = hidden_states

        for t in range(seq_len):
            hidden_states = np.tanh(
                (inputs[:, t] @ self._weights_input_hidden +
                 hidden_states @ self._weights_hidden_hidden +
                 self._hidden_bias))
            self._last_hidden_states[:, t + 1] = hidden_states

        return hidden_states

    def backward(self, final_gradients: np.ndarray) -> np.ndarray:
        if self._last_inputs is None or self._last_hidden_states is None:
            logger.error("Backward pass called before forward pass.")
            raise RuntimeError("Forward pass must be called before backward.")

        if final_gradients.shape != (self._last_inputs.shape[0],
                                     self.hidden_dim):
            logger.error(
                "Final gradients shape mismatch. Expected (%d, %d), got"
                " %s.", self._last_inputs.shape[0], self.hidden_dim,
                final_gradients.shape)
            raise RuntimeError(
                f"Final gradients shape mismatch. Expected ({self._last_inputs.shape[0]}, "
                f"{self.hidden_dim}), got {final_gradients.shape}")

        batch_size, seq_len, _ = self._last_inputs.shape

        if self.gradients_input_hidden is None:
            self.gradients_input_hidden = np.zeros_like(
                self._weights_input_hidden)
        else:
            self.gradients_input_hidden.fill(0.0)

        if self.gradients_hidden_hidden is None:
            self.gradients_hidden_hidden = np.zeros_like(
                self._weights_hidden_hidden)
        else:
            self.gradients_hidden_hidden.fill(0.0)

        if self.gradients_hidden_bias is None:
            self.gradients_hidden_bias = np.zeros_like(self._hidden_bias)
        else:
            self.gradients_hidden_bias.fill(0.0)

        input_sequence_gradients = np.zeros(
            (batch_size, seq_len, self.input_dim))

        hidden_gradients = final_gradients.copy()

        for t in reversed(range(seq_len)):
            current_hidden = self._last_hidden_states[:, t + 1]
            previous_hidden = self._last_hidden_states[:, t]
            current_inputs = self._last_inputs[:, t]

            pre_activation_gradients = hidden_gradients * (1 -
                                                           current_hidden**2)
            self.gradients_input_hidden += current_inputs.T @ pre_activation_gradients
            self.gradients_hidden_hidden += previous_hidden.T @ pre_activation_gradients
            self.gradients_hidden_bias += pre_activation_gradients.sum(axis=0)

            input_gradients = pre_activation_gradients @ self._weights_input_hidden.T
            input_sequence_gradients[:, t] = input_gradients

            hidden_gradients = previous_hidden

        return input_sequence_gradients
