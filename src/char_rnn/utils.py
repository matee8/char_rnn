import logging
from pathlib import Path
from typing import Dict

import numpy as np

from char_rnn.models import Model

logger = logging.getLogger(__name__)


def load_data_from_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found at '{path}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()

        logger.info(
            "Successfully loaded data from '%s'. Total characters: %d.", path,
            len(text_data))

        return text_data
    except IOError as e:
        raise IOError(f"Could not read file at '{path}'.") from e


def save_model_weights(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    weights: Dict[str, np.ndarray] = {}
    logger.debug("Preparing to save weights for model '%s'.", model.name)

    for i, layer in enumerate(model.layers):
        for param_name, param_value in layer.params.items():
            key = f"layer_{i}_{layer.name}_{param_name}"
            weights[key] = param_value
            logger.debug("Preparint to save: '%s' with shape %s.", key,
                         param_value.shape)

    if not weights:
        logger.warning("No trainable parameters found in model '%s'.",
                       model.name)
        return

    np.savez_compressed(path, **weights, allow_pickle=False)
    logger.info("Model weights (%d arrays) successfully saved to '%s'.",
                len(weights), path)


def load_model_weights(model: Model, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Weights file not found at '{path}'.")

    with np.load(path, allow_pickle=False) as weights:
        logger.info("Successfully loaded weights from '%s'.", path)

        loaded_keys = set(weights.keys())
        expected_keys = set()

        logger.debug("Loading weights into model '%s'.", model.name)

        for i, layer in enumerate(model.layers):
            for param_name, current_value in layer.params.items():
                key = f"layer_{i}_{layer.name}_{param_name}"
                expected_keys.add(key)

                if key not in weights:
                    raise ValueError(f"Parameter '{key}' (for layer "
                                     f"'{layer.name}', param '{param_name}') "
                                     f"not found in weights file '{path}'.")

                loaded_params = weights[key]

                if current_value.shape != loaded_params.shape:
                    raise ValueError(f"Shape mismatch for parameter '{key}' "
                                     f"(layer '{layer.name}', param "
                                     f"'{param_name}'). Model expects shape "
                                     f"{current_value.shape} but file contains"
                                     f"{loaded_params.shape}.")

                current_value[:] = loaded_params
                logger.debug(
                    "Loaded weights for '%s' with shape %s into layer "
                    "'%s'.", key, current_value.shape, layer.name)

        extra_keys = loaded_keys - expected_keys
        if extra_keys:
            logger.warning(
                "Weights file '%s' contains extra, unused "
                "parameter arrays: %s.", path, sorted(list(extra_keys)))

        logger.info(
            "All expected model weights successfully loaded into model "
            "%s.", model.name)
