import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Layer(ABC):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._last_x: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def params(self) -> Dict[str, np.ndarray]:
        pass

    @property
    @abstractmethod
    def grads(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dL_dy: np.ndarray) -> Optional[np.ndarray]:
        pass


class Embedding(Layer):

    def __init__(self, V: int, D_e: int, name: Optional[str] = None) -> None:
        super().__init__(name)

        if V <= 0:
            raise ValueError(f"Vocabulary size (V) must be positive, got {V}.")

        if D_e <= 0:
            raise ValueError(
                f"Embedding dimension (D_e) must be positive, got {D_e}.")

        self.V = V
        self.D_e = D_e
        self._W_e = np.random.randn(V, D_e).astype(np.float32) * 0.01
        self._dL_dW_e = np.zeros_like(self._W_e, dtype=np.float32)

        logger.info("%s initialized with V=%d, D_e=%d.", self.name, self.V,
                    self.D_e)

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W_e": self._W_e}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        return {"W_e": self._dL_dW_e}

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("Input shape mismatch. Expected 2D NumPy array, "
                             f"got {x.ndim}D array with shape {x.shape}.")

        self._last_x = x

        try:
            y = self._W_e[x]
        except IndexError as e:
            raise ValueError("Input contains indices out of bounds for "
                             f"vocabulary size {self.V}.") from e

        logger.debug("%s forward pass: input_shape=%s, output_shape=%s.",
                     self.name, x.shape, y.shape)

        return y

    def backward(self, dL_dy: np.ndarray) -> Optional[np.ndarray]:
        if self._last_x is None:
            raise RuntimeError("Must call forward() before backward()")

        if dL_dy.ndim != 3:
            raise ValueError(
                "Output gradients shape mismatch. Expected 3D NumPy array, "
                f"got {dL_dy.ndim}D array with shape{dL_dy.shape}.")

        expected_shape = (*self._last_x.shape, self.D_e)
        if dL_dy.shape != expected_shape:
            raise ValueError(
                "Output gradients shape mismatch. Expected NumPy array with "
                f"shape {expected_shape}, got {dL_dy.shape}.")

        self._dL_dW_e.fill(0.0)

        flat_indices = self._last_x.ravel()
        reshaped_output_gradients = dL_dy.reshape(-1, self.D_e)
        np.add.at(self._dL_dW_e, flat_indices, reshaped_output_gradients)

        logger.debug("%s backward pass: dL_dy_shape=%s.", self.name,
                     dL_dy.shape)

        return None


class Recurrent(Layer):

    def __init__(self,
                 D_in: int,
                 D_h: int,
                 name: Optional[str] = None) -> None:
        super().__init__(name)

        if D_in <= 0:
            raise ValueError(
                f"Input dimension (D_in) must be positive, got {D_in}.")

        if D_h <= 0:
            raise ValueError(
                f"Hidden dimension (D_h) must be positive, got {D_h}.")

        self.D_in = D_in
        self.D_h = D_h

        self._W_xh = np.random.randn(D_in, D_h) * 0.01
        self._W_hh = np.random.randn(D_h, D_h) * 0.01
        self._b_h = np.zeros((1, D_h))

        self._dL_dW_xh = np.zeros_like(self._W_xh)
        self._dL_dW_hh = np.zeros_like(self._W_hh)
        self._dL_db_h = np.zeros_like(self._b_h)

        self._last_x: Optional[np.ndarray] = None
        self._last_h_seq: Optional[np.ndarray] = None

        logger.info("%s initialized with D_in=%d, D_h=%d.", self.name,
                    self.D_in, self.D_h)

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W_xh": self._W_xh, "W_hh": self._W_hh, "b_h": self._b_h}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self._dL_dW_xh,
            "W_hh": self._dL_dW_hh,
            "b_h": self._dL_db_h
        }

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)

        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(1, -1)

        if x_t.shape[1] != self.D_in:
            raise ValueError(
                "Input feature dimension mismatch. Expected shape[1] to be "
                f"{self.D_in}, got {x_t.shape[1]} from shape {x_t.shape}.")

        if h_prev.shape[1] != self.D_h:
            raise ValueError(
                "Hidden state dimension mismatch. Expected shape[1] to be "
                f"{self.D_h}, got {h_prev.shape[1]} from shape {h_prev.shape}."
            )

        if x_t.shape[0] != h_prev.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x_t ({x_t.shape[0]})"
                f" and h_prev ({h_prev.shape[0]}).")

        a_h_t = x_t @ self._W_xh + h_prev @ self._W_hh + self._b_h

        h_t = np.tanh(a_h_t)

        return h_t

    def forward(self,
                x: np.ndarray,
                h_0: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("Input shape mismatch. Expected 3D NumPy array, "
                             f"got {x.ndim}D array with shape {x.shape}.")

        N, T_seq, input_dim = x.shape

        if input_dim != self.D_in:
            raise ValueError(
                "Input feature dimension mismatch. Expected shape[-1] to be "
                f"{self.D_in}, got {x.shape[-1]} from shape {x.shape}.")

        expected_h_0_shape = (N, self.D_h)
        if h_0 is None:
            h_t = np.zeros(expected_h_0_shape)
        else:
            if h_0.shape != expected_h_0_shape:
                raise ValueError("Initial hidden state shape mismatch. "
                                 f"Expected {expected_h_0_shape}, got "
                                 f"{h_0.shape}.")

            h_t = h_0

        self._last_x = x
        self._last_h_seq = np.zeros((N, T_seq + 1, self.D_h), dtype=x.dtype)
        self._last_h_seq[:, 0, :] = h_t

        for t in range(T_seq):
            x_t = x[:, t, :]
            h_t = self.forward_step(x_t, h_t)
            self._last_h_seq[:, t + 1, :] = h_t

        logger.debug("%s forward pass: x_shape=%s, final_h_shape=%s.",
                     self.name, x.shape, h_t.shape)

        return h_t

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        if self._last_x is None or self._last_h_seq is None:
            raise RuntimeError("Must call forward() before backward().")

        if dL_dy.ndim != 2:
            raise ValueError(
                "Output gradient shape mismatch. Expected 2D NumPy array, got "
                f"{dL_dy.ndim}D array with shape {dL_dy.shape}.")

        N, T_seq, _ = self._last_x.shape

        expected_dL_dy_shape = (N, self.D_h)
        if dL_dy.shape != expected_dL_dy_shape:
            raise ValueError("Output gradient shape mismatch. Expected "
                             f"{expected_dL_dy_shape}, got {dL_dy.shape}.")

        self._dL_dW_xh.fill(0.0)
        self._dL_dW_hh.fill(0.0)
        self._dL_db_h.fill(0.0)

        dL_dx_seq = np.zeros_like(self._last_x)

        dL_dh_t = dL_dy.copy()

        for t in reversed(range(T_seq)):
            h_t = self._last_h_seq[:, t + 1, :]
            h_prev_t = self._last_h_seq[:, t, :]
            x_t = self._last_x[:, t, :]

            dtanh_da_h_t = 1 - h_t**2

            dL_da_h_t = dL_dh_t * dtanh_da_h_t

            self._dL_dW_xh += x_t.T @ dL_da_h_t
            self._dL_dW_hh += h_prev_t.T @ dL_da_h_t
            self._dL_db_h += np.sum(dL_da_h_t, axis=0, keepdims=True)

            dL_dx_seq[:, t, :] = dL_da_h_t @ self._W_xh.T

            dL_dh_t = dL_da_h_t @ self._W_hh.T

        logger.debug("%s backward pass: dL_dy_shape=%s, dL_dX_seq_shape=%s.",
                     self.name, dL_dy.shape, dL_dx_seq.shape)

        return dL_dx_seq


class Dense(Layer):

    def __init__(self,
                 D_in: int,
                 D_out: int,
                 name: Optional[str] = None) -> None:
        super().__init__(name)

        if D_in <= 0:
            raise ValueError(
                f"Input dimension (D_in) must be positive, got {D_in}.")

        if D_out <= 0:
            raise ValueError(
                f"Output dimension (D_out) must be positive, got {D_out}.")

        self.D_in = D_in
        self.D_out = D_out

        self._W = np.random.randn(D_in, D_out) * 0.01
        self._b = np.zeros((1, D_out))

        self._dL_dW = np.zeros_like(self._W)
        self._dL_db = np.zeros_like(self._b)

        self._last_p: Optional[np.ndarray] = None

        logger.info("%s initialized with D_in=%d, D_out=%d.", self.name,
                    self.D_in, self.D_out)

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W": self._W, "b": self._b}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        return {"W": self._dL_dW, "b": self._dL_db}

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("Input shape mismatch. Expected 2D NumPy array, "
                             f"got {x.ndim}D array with shape {x.shape}.")

        if x.shape[-1] != self.D_in:
            raise ValueError(
                "Input feature dimension mismatch. Expected shape[-1] to be "
                f"{self.D_in}, got {x.shape[-1]} from shape {x.shape}.")

        self._last_x = x

        z = x @ self._W + self._b
        p = self._softmax(z)

        self._last_p = p

        logger.debug("%s forward pass: input_shape=%s, output_shape=%s.",
                     self.name, x.shape, p.shape)

        return p

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        if self._last_x is None or self._last_p is None:
            raise RuntimeError("Must call forward() before backward().")

        if dL_dy.ndim != 2:
            raise ValueError(
                "Output gradients shape mismatch. Expected 2D NumPy array, "
                f"got {dL_dy.ndim}D array with shape {dL_dy.shape}.")

        if dL_dy.shape != self._last_p.shape:
            raise ValueError("Output gradients shape mismatch. Expected "
                             f"{self._last_p.shape}, got {dL_dy.shape}.")

        s = np.sum(dL_dy * self._last_p, axis=1, keepdims=True)
        dL_dz = self._last_p * (dL_dy - s)

        self._dL_dW = self._last_x.T @ dL_dz
        self._dL_db = np.sum(dL_dz, axis=0, keepdims=True)

        dL_dx = dL_dz @ self._W.T

        logger.debug("%s backward pass: dL_dy_shape=%s, dL_dx_shape=%s.",
                     self.name, dL_dy.shape, dL_dx.shape)

        return dL_dx

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z_stabilized = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stabilized)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
