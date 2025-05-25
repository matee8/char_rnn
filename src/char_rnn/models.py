import logging
from typing import List, Optional

import numpy as np

from char_rnn import preprocessing
from char_rnn.layers import Layer
from char_rnn.losses import Loss
from char_rnn.optimizers import Optimizer

logger = logging.getLogger(__name__)


class Model:

    def __init__(self,
                 layers: List[Layer],
                 name: Optional[str] = None) -> None:
        if not layers:
            raise ValueError("Model must have at least one layer.")

        self.name = name or self.__class__.__name__
        self.layers = layers

        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None

        self._h_t_final: Optional[np.ndarray] = None

        logger.info("%s initialized with %d layers.", self.name,
                    len(self.layers))

    def compile(self, optimizer: Optimizer, loss: Loss) -> None:
        self.optimizer = optimizer
        self.loss = loss

        logger.info("%s compiled with optimizer: %s, loss function: %s.",
                    self.name, self.optimizer.name, self.loss.name)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        logger.debug("Generating predictions for input shape: %s", x.shape)

        return self._forward(x, **kwargs)

    def evaluate(self, x_batches: np.ndarray, y_batches: np.ndarray,
                 **kwargs) -> float:
        if x_batches.shape[0] != y_batches.shape[0]:
            raise ValueError(
                f"Number of input batches ({x_batches.shape[0]}) must match "
                f"number of target batches ({y_batches.shape[0]}).")

        correct = 0
        total = 0
        num_batches = x_batches.shape[0]

        logger.debug("Starting evaluation over %d batches.", num_batches)

        for i, (x, y) in enumerate(zip(x_batches, y_batches)):
            try:
                y_proba = self._forward(x, **kwargs)
                y_pred_labels = np.argmax(y_proba, axis=1)

                correct += np.sum(y_pred_labels == y)

                if (i + 1) % max(1, num_batches // 10) == 0:
                    logger.info(
                        "Evaluation progress: %d/%d batches processed.", i + 1,
                        num_batches)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error evaluating batch %d: %s.",
                             i + 1,
                             e,
                             exc_info=True)
                continue

        if total == 0:
            raise RuntimeError("No samples found in evaluation data after "
                               "iterating batches.")

        accuracy = correct / total

        logger.info("%s evaluation completed. Accuracy: %.4f (%d/%d correct).",
                    self.name, accuracy, correct, total)

        return accuracy

    def fit(self,
            x_batches: np.ndarray,
            y_batches: np.ndarray,
            num_epochs: int,
            log_interval: int,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            seed: Optional[int] = None) -> None:
        if self.optimizer is None or self.loss is None:
            raise RuntimeError(
                f"{self.name} has not been compiled yet. Call compile("
                "optimizer, loss) before training.")

        if num_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")

        if log_interval <= 0:
            raise ValueError("Log interval must be positive.")

        if x_batches.shape[0] != y_batches.shape[0]:
            raise ValueError(
                f"Number of input batches ({x_batches.shape[0]}) must match "
                f"number of target batches ({y_batches.shape[0]}).")

        num_batches = x_batches.shape[0]
        if num_batches == 0:
            logger.warning(
                "Training data (x_batches) is empty. Skipping training.")
            return

        if x_val is not None and y_val is not None:
            if x_val.shape[0] > 0:
                logger.info("Validation dataset provided with %d batches.",
                            x_val.shape[0])
            else:
                logger.warning("Validation dataset is empty. Skipping "
                               "validation.")

        logger.info(
            "Starting training for %s: %d epochs, %d batches per epoch.",
            self.name, num_epochs, num_batches)

        for epoch in range(1, num_epochs + 1):
            epoch_losses: List[float] = []

            current_seed = (seed + epoch - 1) if seed is not None else None

            x_shuffled, y_shuffled = preprocessing.shuffle_batches(
                x_batches, y_batches, current_seed)

            for i, (x, y) in enumerate(zip(x_shuffled, y_shuffled)):
                try:
                    y_proba = self._forward(x)

                    loss = self.loss.forward(y_proba, y)
                    epoch_losses.append(loss)

                    dL_dy_proba = self.loss.backward(y_proba, y)
                    self._backward(dL_dy_proba)

                    self.optimizer.step(self.layers)

                    if (i + 1) % log_interval == 0:
                        logger.info("Epoch %d/%d - Batch %d/%d - Loss: %.4f",
                                    epoch, num_epochs, i + 1, num_batches,
                                    loss)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error training batch %d in epoch %d: %s",
                                 i + 1,
                                 epoch,
                                 e,
                                 exc_info=True)
                    continue

            avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info("Epoch %d/%d - Average Training Loss: %.4f", epoch,
                        num_epochs, avg_loss)
            if x_val is not None and y_val is not None and x_val.shape[0] > 0:
                try:
                    val_accuracy = self.evaluate(x_val, y_val)
                    logger.info("Epoch %d/%d - Validation accuracy: %.4f",
                                epoch, num_epochs, val_accuracy)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error during validation for epoch %d: %s",
                                 epoch,
                                 e,
                                 exc_info=True)

        logger.info("Training finished for model %s.", self.name)

    def _forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        z = x

        for layer in self.layers:
            z = layer.forward(z, **kwargs)

        return z

    def _backward(self, dL_dy: np.ndarray) -> None:
        gradient = dL_dy

        for i, layer in enumerate(reversed(self.layers)):
            gradient = layer.backward(
                gradient)  # type: ignore[reportArgumentType]
            if gradient is None and i != len(self.layers) - 1:
                raise RuntimeError(
                    f"Layer {layer.name} backward pass returned "
                    "None unexpectedly.")
