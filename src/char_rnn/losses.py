import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Loss(ABC):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class SparseCategoricalCrossEntropy(Loss):

    def __init__(self, eps: float = 1e-12, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.eps = eps

        logger.info("%s initialized with epsilon=%.0e.", self.name, self.eps)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_pred.ndim != 2:
            raise ValueError(
                "Predictions ndim mismatch. Expected 2D NumPy array, got "
                f"{y_pred.ndim}D array with shape {y_pred.shape}.")

        if y_true.ndim != 1:
            raise ValueError(
                "True outputs ndim mismatch. Expected 1D NumPy array, got "
                f"{y_true.ndim}D array with shape {y_true.shape}.")

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Batch size mismatch between predictions ({y_pred.shape[0]}) "
                f"and true outputs ({y_true.shape[0]}).")

        N, V = y_pred.shape

        if np.any(y_true < 0) or np.any(y_true >= V):
            raise ValueError(
                "True outputs contains values out of bounds for number of "
                f"classes {V}.")

        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)

        p_true = y_pred_clipped[np.arange(N), y_true]

        log_probas = -np.log(p_true)

        mean_loss = np.mean(log_probas)

        logger.debug(
            "%s forward pass: y_pred_shape=%s, y_true_shape=%s, "
            "loss=%.4f", self.name, y_pred.shape, y_true.shape, mean_loss)

        return float(mean_loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if y_pred.ndim != 2:
            raise ValueError(
                "Predictions ndim mismatch. Expected 2D NumPy array, got "
                f"{y_pred.ndim}D array with shape {y_pred.shape}.")

        if y_true.ndim != 1:
            raise ValueError(
                "True outputs ndim mismatch. Expected 1D NumPy array, got "
                f"{y_true.ndim}D array with shape {y_true.shape}.")

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Batch size mismatch between predictions ({y_pred.shape[0]}) "
                f"and true outputs ({y_true.shape[0]}).")

        N = y_pred.shape[0]

        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)

        dL_dp = np.zeros_like(y_pred_clipped)

        dL_dp[np.arange(N),
              y_true] = -1.0 / y_pred_clipped[np.arange(N), y_true]

        dL_dp /= N

        logger.debug(
            "%s backward pass: y_pred_shape=%s, y_true_shape=%s, "
            "dL_dp_shape=%s.", self.name, y_pred.shape, y_true.shape,
            dL_dp.shape)

        return dL_dp
