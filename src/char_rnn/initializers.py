import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Initializer(ABC):

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def initialize(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype("float32")
    ) -> np.ndarray:
        pass


class Zeros(Initializer):

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        logger.info("%s initializer created.", self.name)

    def initialize(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype("float32")
    ) -> np.ndarray:
        weights = np.zeros(shape, dtype=dtype)

        logger.info("%s initialized weights with shape %s to zeros.",
                    self.name, shape)
        return weights


class RandomUniform(Initializer):

    def __init__(self,
                 minval: float = -0.05,
                 maxval: float = 0.05,
                 seed: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(name)

        if minval >= maxval:
            raise ValueError("Minimum value must be less than maximum value")

        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        logger.info(
            "%s initializer created with minval=%.4f, maxval=%.4f, "
            "seed=%s.", self.name, self.minval, self.maxval,
            self.seed if self.seed is not None else "None")

    def initialize(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype("float32")
    ) -> np.ndarray:
        weights = (self._rng.uniform(self.minval, self.maxval,
                                     shape).astype(dtype))

        logger.info(
            "%s initialized weights with shape %s from uniform "
            "distribution.", self.name, shape)

        return weights


class GlorotUniform(Initializer):

    def __init__(self,
                 seed: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(name)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        logger.info("%s initializer created with seed=%s.", self.name,
                    self.seed if self.seed is not None else "None")

    def initialize(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype("float32")
    ) -> np.ndarray:
        if len(shape) < 2:
            raise ValueError("GlorotUniform requires at least 2D shape, "
                             f"got {len(shape)}D.")

        fan_in = shape[-2]
        fan_out = shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        weights = self._rng.uniform(-limit, limit, shape).astype(dtype)

        logger.info(
            "%s initialized weights with shape %s, fan_in=%d, "
            "fan_out=%d, limit=%.4f.", self.name, shape, fan_in, fan_out,
            limit)

        return weights


class Orthogonal(Initializer):

    def __init__(self,
                 gain: float = 1.0,
                 seed: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(name)

        self.gain = gain
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        logger.info("%s initializer created with gain=%.4f, seed=%s.",
                    self.name, self.gain,
                    self.seed if self.seed is not None else "None")

    def initialize(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype("float32")
    ) -> np.ndarray:
        if len(shape) < 2:
            raise ValueError("Orthogonal requires at least 2D shape, "
                             f"got {len(shape)}D.")

        effective_shape = (int(np.prod(shape[:-1])), shape[-1])

        if effective_shape[0] < effective_shape[1]:
            svd_shape = (effective_shape[1], effective_shape[0])
        else:
            svd_shape = effective_shape

        a = self._rng.normal(0.0, 1.0, size=svd_shape).astype(dtype)
        u, _, _ = np.linalg.svd(a, full_matrices=False)

        q_ortho = u

        if effective_shape[0] < effective_shape[1]:
            q_ortho = q_ortho.T

        if len(shape) > 2:
            if q_ortho.size != np.prod(shape):
                raise ValueError("Cannot reshape orthogonal matrix of size "
                                 f"{q_ortho.size} to {np.prod(shape)}")

            q_ortho = q_ortho.reshape(shape)

        weights = self.gain * q_ortho

        logger.info("%s initialized weights with shape %s, gain=%.4f.",
                    self.name, shape, self.gain)

        return weights
