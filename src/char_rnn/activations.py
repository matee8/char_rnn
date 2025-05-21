import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Activation(ABC):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dL_dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class Tanh(Activation):

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        logger.info("%s activation function initialized.", self.name)

    def forward(self, z: np.ndarray) -> np.ndarray:
        y = np.tanh(z)

        logger.debug("%s forward pass: input_shape=%s, output_shape=%s.",
                     self.name, z.shape, y.shape)

        return y

    def backward(self, dL_dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        if dL_dy.shape != y.shape:
            raise ValueError(
                f"Shape mismatch between output gradients ({dL_dy.shape}) and "
                f"output ({y.shape}). They must be identical.")

        dtanh_dz = 1 - y**2
        dL_dz = dL_dy * dtanh_dz

        logger.debug(
            "%s backward pass: dL_dy_shape=%s, y_shape=%s, "
            "dL_dz_shape=%s.", self.name, dL_dy.shape, y.shape, dL_dz.shape)

        return dL_dz


class Softmax(Activation):

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        logger.info("%s activation function initialized.", self.name)

    def forward(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2:
            raise ValueError("Input 'z' must be a 2D NumPy array for Softmax, "
                             f"got {z.ndim}D array with shape {z.shape}.")

        z_stabilized = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stabilized)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        logger.debug("%s forward pass: input_shape=%s, output_shape=%s.",
                     self.name, z.shape, y.shape)

        return y

    def backward(self, dL_dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        if dL_dy.shape != y.shape:
            raise ValueError(
                f"Shape mismatch between output gradients ({dL_dy.shape}) and "
                f"output ({y.shape}). They must be identical.")

        s = np.sum(dL_dy * y, axis=1, keepdims=True)
        dL_dz = y * (dL_dy - s)

        logger.debug(
            "%s backward pass: dL_dy_shape=%s, y_shape=%s, "
            "dL_dz_shape=%s.", self.name, dL_dy.shape, y.shape, dL_dz.shape)

        return dL_dz
