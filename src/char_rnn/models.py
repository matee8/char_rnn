import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from char_rnn.layers import Dense, Embedding, Layer, Recurrent
from char_rnn.losses import Loss
from char_rnn.optimizers import Optimizer
from char_rnn.preprocessing import TextVectorizer

logger = logging.getLogger(__name__)


class Model(ABC):

    def __init__(self, name: Optional[str]) -> None:
        self.name = name or self.__class__.__name__

        self.trainable_layers: List[Layer] = []

        self._loss_fn: Optional[Loss] = None
        self._optimizer: Optional[Optimizer] = None

    @abstractmethod
    def train_step(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def compile(self, optimizer: Optimizer, loss_fn: Loss) -> None:
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        logger.info("%s compiled with optimizer: %s, loss function: %s.",
                    self.name, self._optimizer.name, self._loss_fn.name)


class CharRNN(Model):

    def __init__(self,
                 V: int,
                 D_e: int,
                 D_h: int,
                 name: Optional[str] = None) -> None:
        super().__init__(name)

        if V <= 0:
            raise ValueError("Vocabulary size (V) must be positive.")

        if D_e <= 0:
            raise ValueError("Embedding dimension (D_e) must be positive.")

        if D_h <= 0:
            raise ValueError("Hidden dimension (D_h) must be positive.")

        self.V = V
        self.D_e = D_e
        self.D_h = D_h

        self._embedding_layer = Embedding(V=self.V, D_e=self.D_e)
        self._recurrent_layer = Recurrent(D_in=self.D_e, D_h=self.D_h)
        self._dense_layer = Dense(D_in=self.D_h, D_out=self.V)

        self.trainable_layers = [
            self._embedding_layer, self._recurrent_layer, self._dense_layer
        ]

        self._h_t_final: Optional[np.ndarray] = None

        logger.info("%s initialized with V=%d, D_e=%d, D_h=%d", self.name,
                    self.V, self.D_e, self.D_h)

    def train_step(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        if self._loss_fn is None or self._optimizer is None:
            raise RuntimeError(
                f"{self.name} has not been compiled yet. Call compile("
                "optimizer, loss_fn) before training or evaluation.")

        y_pred = self._forward_pass(x)

        loss = self._loss_fn.forward(y_pred, y)

        dL_dy_pred = self._loss_fn.backward(y_pred, y)

        self._backward_pass(dL_dy_pred)

        self._optimizer.step(self.trainable_layers)

        return loss

    def predict(self,
                x: np.ndarray,
                h_0: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if h_0 is not None and h_0.ndim == 1:
            h_0 = h_0.reshape(1, -1)

        return self._forward_pass(x, h_0)

    def predict_next_char(self,
                          x: np.ndarray,
                          h_0: Optional[np.ndarray] = None,
                          temp: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if temp <= 0.0:
            raise ValueError("Temperature must be positive.")

        y_proba = self.predict(x, h_0)

        if self._h_t_final is None:
            raise RuntimeError(
                "Last hidden state is None after forward pass unexpectedly.")

        if temp == 1.0:
            char_id = np.argmax(y_proba, axis=1)
        else:
            y_proba_clipped = np.clip(y_proba, 1e-9, 1.0)

            log_probas = np.log(y_proba_clipped)
            scaled_log_probas = log_probas / temp

            exp_scaled_log_probas = (
                np.exp(scaled_log_probas -
                       np.max(scaled_log_probas, axis=1, keepdims=True)))

            final_probas = (
                exp_scaled_log_probas /
                np.sum(exp_scaled_log_probas, axis=1, keepdims=True))

            char_id = np.array([
                np.random.choice(self.V, p=final_probas[i])
                for i in range(y_proba.shape[0])
            ])

        return char_id, self._h_t_final

    def generate_sequence(self,
                          vectorizer: TextVectorizer,
                          text: str,
                          n_chars: int,
                          temperature: float = 1.0) -> str:
        if self.V != vectorizer.vocabulary_size:
            logger.error(
                "%s: model vocab size (%d) does not match "
                "vectorizer vocab size (%d). Ensure they are "
                "compatible.", self.name, self.V, vectorizer.vocabulary_size)

        logger.debug(
            "%s generating sequence: start_string='%s', num_chars=%d, "
            "temp=%.2f.", self.name, text, n_chars, temperature)

        generated_text = text

        x = vectorizer.encode([text])[0]

        h_t = np.zeros((1, self.D_h))

        for _ in range(n_chars):
            x = x.reshape(1, -1)

            next_char_idx_array, h_t = self.predict_next_char(
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
        self._h_t_final = h_t_final

        return self._dense_layer.forward(h_t_final)

    def _backward_pass(self, dL_dy_pred: np.ndarray) -> None:
        dL_dh_t_final = self._dense_layer.backward(dL_dy_pred)

        if dL_dh_t_final is None:
            raise RuntimeError(
                "Dense layer backward pass returned None unexpectedly.")

        dL_dy_emb = self._recurrent_layer.backward(dL_dh_t_final)
        if dL_dy_emb is None:
            raise RuntimeError(
                "Recurrent layer backward pass returned None unexpectedly.")

        self._embedding_layer.backward(dL_dy_emb)
