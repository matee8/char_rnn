from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        pass
