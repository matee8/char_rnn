import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from char_rnn import preprocessing
from char_rnn.layers import Dense, Embedding, Layer, Recurrent
from char_rnn.losses import Loss
from char_rnn.optimizers import Optimizer

logger = logging.getLogger(__name__)


class Model(ABC):

    def __init__(self, name: Optional[str]) -> None:
        self.name = name or self.__class__.__name__

        self.trainable_layers: List[Layer] = []

        self._loss_fn: Optional[Loss] = None
        self._optimizer: Optional[Optimizer] = None

    def compile(self, optimizer: Optimizer, loss_fn: Loss) -> None:
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        logger.info("%s compiled with optimizer: %s, loss function: %s.",
                    self.name, self._optimizer.name, self._loss_fn.name)

    @abstractmethod
    def _forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self, dL_dy: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray,
                **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass

    def fit(self,
            X_batches: np.ndarray,
            y_batches: np.ndarray,
            num_epochs: int,
            log_interval: int,
            seed: Optional[int] = None) -> None:
        if self._loss_fn is None or self._optimizer is None:
            raise RuntimeError(
                f"{self.name} has not been compiled yet. Call compile("
                "optimizer, loss_fn) before training or evaluation.")

        if num_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")

        if log_interval <= 0:
            raise ValueError("Interval of logs must be positive.")

        num_total_batches = X_batches.shape[0]

        logger.info(
            "Starting training for %s over %d epochs, with %d batches "
            "per epoch.", self.name, num_epochs, num_total_batches)

        for epoch in range(1, num_epochs + 1):
            epoch_losses: List[float] = []

            X_shuffled, y_shuffled = (preprocessing.shuffle_batches(
                X_batches, y_batches,
                (seed + epoch if seed is not None else None)))

            for i, (x, y) in enumerate(zip(X_shuffled, y_shuffled)):
                try:
                    y_pred = self._forward(x)

                    loss = self._loss_fn.forward(y_pred, y)
                    epoch_losses.append(loss)

                    dL_dy_pred = self._loss_fn.backward(y_pred, y)
                    self._backward(dL_dy_pred)

                    self._optimizer.step(self.trainable_layers)

                    if (i + 1) % log_interval == 0:
                        logger.info("Epoch %d/%d - Batch %d/%d - Loss: %.4f",
                                    epoch, num_epochs, i + 1,
                                    num_total_batches, loss)
                except Exception as e: # pylint: disable=broad-exception-caught
                    logger.error(
                        "Error during training step for batch %d "
                        "in epoch %d: %s.",
                        i + 1,
                        epoch,
                        e,
                        exc_info=True)
                    continue

            if not epoch_losses:
                logger.warning("Epoch %d completed with no batches processed.")
                avg_epoch_loss = float("nan")
            else:
                avg_epoch_loss = np.mean(epoch_losses)

            logger.info("Epoch %d/%d - Average Loss: %.4f", epoch, num_epochs,
                        avg_epoch_loss)


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

    def _forward(self,
                 x: np.ndarray,
                 h_0: Optional[np.ndarray] = None,
                 **kwargs) -> np.ndarray:
        y_emb = self._embedding_layer.forward(x)

        h_t_final = self._recurrent_layer.forward(y_emb, h_0=h_0)
        self._h_t_final = h_t_final

        return self._dense_layer.forward(h_t_final)

    def _backward(self, dL_dy: np.ndarray) -> None:
        dL_dh_t_final = self._dense_layer.backward(dL_dy)

        if dL_dh_t_final is None:
            raise RuntimeError(
                "Dense layer backward pass returned None unexpectedly.")

        dL_dy_emb = self._recurrent_layer.backward(dL_dh_t_final)
        if dL_dy_emb is None:
            raise RuntimeError(
                "Recurrent layer backward pass returned None unexpectedly.")

        self._embedding_layer.backward(dL_dy_emb)

    def predict(self,
                x: np.ndarray,
                h_0: Optional[np.ndarray] = None,
                return_hidden: bool = False,
                **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if h_0 is not None and h_0.ndim == 1:
            h_0 = h_0.reshape(1, -1)

        y_pred = self._forward(x, h_0)

        if return_hidden:
            if self._h_t_final is None:
                raise RuntimeError("Expected final hidden states to be set "
                                   "after forward pass but it is None.")
            return y_pred, self._h_t_final
        else:
            return y_pred, None
