import logging
from pathlib import Path
from typing import Dict

import numpy as np

from char_rnn.models import Model

logger = logging.getLogger(__name__)


def load_data_from_file(filepath: str) -> str:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at '{filepath}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()

        logger.info(
            "Successfully loaded data from '%s'. Total characters: %d.",
            filepath, len(text_data))

        return text_data
    except IOError as e:
        raise IOError(f"Could not read file at '{filepath}'.") from e


def save_model_weights(model: Model, filepath: str) -> None:
    path = Path(filepath)

    path.parent.mkdir(parents=True, exist_ok=True)

    weights_to_save: Dict[str, np.ndarray] = {}
    logger.debug("Preparing to save weights for model '%s'.", model.name)

    for i, layer in enumerate(model.trainable_layers):
        for param_name, param_value in layer.params.items():
            key = f"layer_{i}_{layer.name}_{param_name}"
            weights_to_save[key] = param_value
            logger.debug("Preparint to save: '%s' with shape %s.", key,
                         param_value.shape)

    if not weights_to_save:
        logger.warning("No trainable parameters found in model '%s'.",
                       model.name)
        return

    np.savez_compressed(path, **weights_to_save, allow_pickle=False)
    logger.info("Model weights (%d arrays) successfully saved to '%s'.",
                len(weights_to_save), filepath)


def load_model_weights(model: Model, filepath: str) -> None:
    path = Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"Weights file not found at '{filepath}'.")

    with np.load(path, allow_pickle=False) as loaded_weights:
        logger.info("Successfully loaded weights from '%s'.", filepath)

        loaded_keys_in_file = set(loaded_weights.keys())
        expected_keys = set()

        logger.debug("Loading weights into model '%s'.", model.name)

        for i, layer in enumerate(model.trainable_layers):
            for param_name, current_params in layer.params.items():
                key = f"layer_{i}_{layer.name}_{param_name}"
                expected_keys.add(key)

                if key not in loaded_weights:
                    raise ValueError(
                        f"Parameter '{key}' (for layer "
                        f"'{layer.name}', param '{param_name}') "
                        f"not found in weights file '{filepath}'.")

                loaded_params = loaded_weights[key]

                if current_params.shape != loaded_params.shape:
                    raise ValueError(
                        f"Shape mismatch for parameter '{key}' "
                        f"(layer '{layer.name}', param "
                        f"'{param_name}'). Model expects shape "
                        f"{current_params.shape} but file contains"
                        f"{loaded_params.shape}.")

                current_params[:] = loaded_params
                logger.debug(
                    "Loaded weights for '%s' with shape %s into layer "
                    "'%s'.", key, current_params.shape, layer.name)

        extra_keys = loaded_keys_in_file - expected_keys
        if extra_keys:
            logger.warning(
                "Weights file '%s' contains extra, unused "
                "parameter arrays: %s.", filepath, sorted(list(extra_keys)))

        logger.info(
            "All expected model weights successfully loaded into model "
            "%s.", model.name)
