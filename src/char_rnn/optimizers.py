import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from char_rnn.layers import Layer

logger = logging.getLogger(__name__)


class Optimizer(ABC):

    def __init__(self, eta: float, name: Optional[str] = None) -> None:
        if eta <= 0:
            raise ValueError("Learning rate must be positive.")

        self.eta = eta
        self.name = name or self.__class__.__name__

        self.t = 0

    @abstractmethod
    def step(self, layers: List[Layer]) -> None:
        pass


class Adam(Optimizer):

    def __init__(self,
                 eta: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7,
                 name: Optional[str] = None) -> None:
        super().__init__(eta, name)

        if not 0.0 < beta1 < 1.0:
            raise ValueError("beta1 must be between 0 and 1.")

        if not 0.0 < beta2 < 1.0:
            raise ValueError("beta2 must be between 0 and 1.")

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}

        logger.info(
            "%s initialized with eta=%.0e, beta1=%.3f, "
            "beta2=%.3f, epsilon=%.0e.", self.name, self.eta, self.beta1,
            self.beta2, self.epsilon)

    def step(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("No layers provided to update.")

        self._initialize_moments(layers)
        self.t += 1

        for layer in layers:
            layer_id = id(layer)
            if not layer.params:
                logger.warning(
                    "Layer '%s' has no parameters to update. Skipping.",
                    layer.name)
                continue

            for param_name, param_value in layer.params.items():
                grad = layer.grads.get(param_name)
                if grad is None:
                    logger.warning(
                        "Gradient for parameter '%s' in layer "
                        "'%s' not found. Skipping.", param_name, layer.name)
                    continue

                if param_value.shape != grad.shape:
                    raise ValueError(
                        f"Shape mismatch between parameter '{param_name}' "
                        f"({param_value.shape}) and its gradient "
                        f"({grad.shape}) in layer '{layer.name}'.")

                self.m[layer_id][param_name] = (
                    self.beta1 * self.m[layer_id][param_name] +
                    (1 - self.beta1) * grad)
                self.v[layer_id][param_name] = (
                    self.beta2 * self.v[layer_id][param_name] +
                    (1 - self.beta2) * (grad**2))
                m_hat = self.m[layer_id][param_name] / (1 - self.beta1**self.t)
                v_hat = self.v[layer_id][param_name] / (1 - self.beta2**self.t)

                param_value -= self.eta * m_hat / (np.sqrt(v_hat) +
                                                   self.epsilon)

                logger.debug("%s updated param '%s' in layer '%s'.", self.name,
                             param_name, layer.name)

        logger.debug("%s step %d completed. Parameters updated.", self.name,
                     self.t)

    def _initialize_moments(self, layers: List[Layer]) -> None:
        for layer in layers:
            layer_id = id(layer)
            if layer_id not in self.m:
                self.m[layer_id] = {}
                self.v[layer_id] = {}
                for param_name, param_value in layer.params.items():
                    self.m[layer_id][param_name] = np.zeros_like(param_value)
                    self.v[layer_id][param_name] = np.zeros_like(param_value)
                    logger.debug(
                        "%s initialized Adam moments for param '%s' "
                        "in layer '%s'.", self.name, param_name, layer.name)


class Nadam(Optimizer):

    def __init__(self,
                 eta: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7,
                 schedule_decay: float = 0.004,
                 name: Optional[str] = None) -> None:
        super().__init__(eta, name)

        if not 0.0 < beta1 < 1.0:
            raise ValueError("beta1 must be between 0 and 1.")

        if not 0.0 < beta2 < 1.0:
            raise ValueError("beta2 must be between 0 and 1.")

        if schedule_decay < 0.0:
            raise ValueError("schedule_decay must be non-negative.")

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}

        self.mu_cache = {}

        logger.info(
            "%s initialized with eta=%.0e, beta1=%.3f, "
            "beta2=%.3f, epsilon=%.0e, schedule_decay=%.3f.", self.name,
            self.eta, self.beta1, self.beta2, self.epsilon,
            self.schedule_decay)

    def step(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("No layers provided to update.")

        self._initialize_moments(layers)
        self.t += 1

        mu_t = self._get_momentum_schedule(self.t)
        mu_t_plus_1 = self._get_momentum_schedule(self.t + 1)

        for layer in layers:
            layer_id = id(layer)
            if not layer.params:
                logger.warning(
                    "Layer '%s' has no parameters to update. Skipping.",
                    layer.name)
                continue

            for param_name, param_value in layer.params.items():
                grad = layer.grads.get(param_name)
                if grad is None:
                    logger.warning(
                        "Gradient for parameter '%s' in layer "
                        "'%s' not found. Skipping.", param_name, layer.name)
                    continue

                if param_value.shape != grad.shape:
                    raise ValueError(
                        f"Shape mismatch between parameter '{param_name}' "
                        f"({param_value.shape}) and its gradient "
                        f"({grad.shape}) in layer '{layer.name}'.")

                self.m[layer_id][param_name] = (
                    self.beta1 * self.m[layer_id][param_name] +
                    (1 - self.beta1) * grad)

                self.v[layer_id][param_name] = (
                    self.beta2 * self.v[layer_id][param_name] +
                    (1 - self.beta2) * (grad**2))

                m_hat = self.m[layer_id][param_name] / (1 - self.beta1**self.t)

                v_hat = self.v[layer_id][param_name] / (1 - self.beta2**self.t)

                m_bar = (mu_t_plus_1 * m_hat + (1 - mu_t) * grad /
                         (1 - self.beta1**self.t))

                param_value -= self.eta * m_bar / (np.sqrt(v_hat) +
                                                   self.epsilon)

                logger.debug("%s updated param '%s' in layer '%s'.", self.name,
                             param_name, layer.name)

        logger.debug(
            "%s step %d completed. Parameters updated with "
            "mu_t=%.6f, mu_t+1=%.6f.", self.name, self.t, mu_t, mu_t_plus_1)

    def _initialize_moments(self, layers: List[Layer]) -> None:
        for layer in layers:
            layer_id = id(layer)
            if layer_id not in self.m:
                self.m[layer_id] = {}
                self.v[layer_id] = {}
                for param_name, param_value in layer.params.items():
                    self.m[layer_id][param_name] = np.zeros_like(param_value)
                    self.v[layer_id][param_name] = np.zeros_like(param_value)
                    logger.debug(
                        "%s initialized Nadam moments for param '%s' "
                        "in layer '%s'.", self.name, param_name, layer.name)

    def _get_momentum_schedule(self, t: int) -> float:
        if t in self.mu_cache:
            return self.mu_cache[t]

        mu_t = self.beta1 * (1 - 0.5 * (0.96**(t * self.schedule_decay)))

        self.mu_cache[t] = mu_t

        return mu_t
