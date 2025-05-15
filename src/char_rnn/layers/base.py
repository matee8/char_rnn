from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        pass
