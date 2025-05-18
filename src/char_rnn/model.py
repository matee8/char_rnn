import logging
from typing import Optional, Tuple

import numpy as np

from char_rnn.layers import Dense, Embedding, Recurrent
from char_rnn.loss import Loss
from char_rnn.optimizers import Optimizer
from char_rnn.preprocessing import TextVectorizer

logger = logging.getLogger(__name__)


class CharRNN:

    def __init__(self,
                 V: int,
                 D_e: int,
                 D_h: int,
                 name: Optional[str] = None) -> None:
        if V <= 0:
            raise ValueError("Vocabulary size (V) must be positive.")

        if D_e <= 0:
            raise ValueError("Embedding dimension (D_e) must be positive.")

        if D_h <= 0:
            raise ValueError("Hidden dimension (D_h) must be positive.")

        self.name = name or self.__class__.__name__
        self.V = V
        self.D_e = D_e
        self.D_h = D_h

        self._embedding_layer = Embedding(V=self.V, D_e=self.D_e)
        self._recurrent_layer = Recurrent(D_in=self.D_e, D_h=self.D_h)
        self._dense_layer = Dense(D_in=self.D_h, D_out=self.V)

        self._trainable_layers = [
            self._embedding_layer, self._recurrent_layer, self._dense_layer
        ]

        self._loss_fn: Optional[Loss] = None
        self._optimizer: Optional[Optimizer] = None

        self._h_t_final_for_generation: Optional[np.ndarray] = None

        logger.info("%s initialized with V=%d, D_e=%d, D_h=%d", self.name,
                    self.V, self.D_e, self.D_h)

    def compile(self, optimizer: Optimizer, loss_fn: Loss) -> None:
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        logger.info("%s compiled with optimizer: %s, loss function: %s.",
                    self.name, self._optimizer.name, self._loss_fn.name)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        if self._loss_fn is None or self._optimizer is None:
            raise RuntimeError(
                f"{self.name} has not been compiled yet. Call compile("
                "optimizer, loss_fn) before training or evaluation.")

        p_pred = self._forward_pass(x)

        loss_value = self._loss_fn.forward(p_pred, y)

        dL_dp_pred = self._loss_fn.backward(p_pred, y)

        self._backward_pass(dL_dp_pred)

        self._optimizer.step(self._trainable_layers)

        return loss_value

    def predict_next_char_probs(
            self,
            x: np.ndarray,
            h_0: Optional[np.ndarray] = None) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if h_0 is not None and h_0.ndim == 1:
            h_0 = h_0.reshape(1, -1)

        return self._forward_pass(x, h_0)

    def predict_next_char_index(
            self,
            x: np.ndarray,
            h_0: Optional[np.ndarray] = None,
            temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if temperature <= 0.0:
            raise ValueError("Temperature must be positive.")

        p_pred = self.predict_next_char_probs(x, h_0)

        if self._h_t_final_for_generation is None:
            raise RuntimeError(
                "Last hidden state is None after forward pass unexpectedly.")

        if temperature == 1.0:
            next_char_indices = np.argmax(p_pred, axis=1)
        else:
            scaled_logits = np.log(p_pred + 1e+9) / temperature

            exp_logits = np.exp(scaled_logits -
                                np.max(scaled_logits, axis=1, keepdims=True))

            probs_temp_scaled = exp_logits / np.sum(
                exp_logits, axis=1, keepdims=True)

            next_char_indices = np.array([
                np.random.choice(self.V, p=probs_temp_scaled[i])
                for i in range(p_pred.shape[0])
            ])

        return next_char_indices, self._h_t_final_for_generation

    def generate_sequence(self,
                          vectorizer: TextVectorizer,
                          start_string: str,
                          num_chars_to_generate: int,
                          temperature: float = 1.0) -> str:
        if self.V != vectorizer.vocabulary_size:
            logger.warning(
                "%s: model vocab size (%d) does not match "
                "vectorizer vocab size (%d). Ensure they are "
                "compatible.", self.name, self.V, vectorizer.vocabulary_size)

        logger.debug(
            "%s generating sequence: start_string='%s', num_chars=%d, "
            "temp=%.2f.", self.name, start_string, num_chars_to_generate,
            temperature)

        generated_text = start_string

        x = vectorizer.encode([start_string])[0]

        h_t = np.zeros((1, self.D_h))

        for _ in range(num_chars_to_generate):
            x = x.reshape(1, -1)

            next_char_idx_array, h_t = self.predict_next_char_index(
                x, h_t, temperature)

            next_char_idx = next_char_idx_array[0]

            next_char = vectorizer.decode(np.array([[next_char_idx]]))[0]
            generated_text += next_char

            x = np.array([next_char_idx])

        logger.debug("%s generated sequence: '%s'.", self.name, generated_text)

        return generated_text

    def _forward_pass(self,
                      x: np.ndarray,
                      h_0: Optional[np.ndarray] = None) -> np.ndarray:
        y_emb = self._embedding_layer.forward(x)

        h_t_final = self._recurrent_layer.forward(y_emb, h_0=h_0)
        self._h_t_final_for_generation = h_t_final

        return self._dense_layer.forward(h_t_final)

    def _backward_pass(self, dL_dp_pred: np.ndarray) -> None:
        dL_dh_t_final = self._dense_layer.backward(dL_dp_pred)

        if dL_dh_t_final is None:
            raise RuntimeError(
                "Dense layer backward pass returned None unexpectedly.")

        dL_dy_emb = self._recurrent_layer.backward(dL_dh_t_final)
        if dL_dy_emb is None:
            raise RuntimeError(
                "Recurrent layer backward pass returned None unexpectedly.")

        self._embedding_layer.backward(dL_dy_emb)
