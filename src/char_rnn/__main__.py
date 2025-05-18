#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging
from typing import List

import numpy as np

from char_rnn.preprocessing import TextVectorizer
from char_rnn.layers import Dense, Embedding, Layer, Recurrent
from char_rnn.loss import SparseCategoricalCrossEntropy
from char_rnn.optimizers import Adam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def main():
    try:
        vectorizer = TextVectorizer()

        alphabet: str = "abcdefghijklmnopqrstuvwxyz "
        vectorizer.fit(alphabet)

        texts: List[str] = ["hello", "world"]

        x_indices_full = vectorizer.encode(texts)

        x_indices_input = x_indices_full[:, :-1]
        y_true_indices = x_indices_full[:, -1]
    except ValueError as e:
        logger.error("Error during text vectorization: %s.", e, exc_info=True)
        return
    except RuntimeError as e:
        logger.error("Runtime error during text vectorization: %s.",
                     e,
                     exc_info=True)
        return

    D_e = 16
    D_h = 128
    num_epochs = 1000
    V = vectorizer.vocabulary_size

    try:
        embedding_layer = Embedding(V, D_e)
        recurrent_layer = Recurrent(D_e, D_h)
        dense_layer = Dense(D_h, V)
        loss_fn = SparseCategoricalCrossEntropy()
        optimizer = Adam()

        trainable_layers: List[Layer] = [
            embedding_layer, recurrent_layer, dense_layer
        ]

        for epoch in range(num_epochs):
            y_emb = embedding_layer.forward(x_indices_input)
            h_t_final = recurrent_layer.forward(y_emb)
            p = dense_layer.forward(h_t_final)

            mean_loss = loss_fn.forward(p, y_true_indices)
            logger.info("Epoch %d - Loss: %.4f.", epoch + 1, mean_loss)

            dL_dp = loss_fn.backward(p, y_true_indices)

            dL_dh_t_final = dense_layer.backward(dL_dp)
            dL_dy_emb = recurrent_layer.backward(dL_dh_t_final)
            embedding_layer.backward(dL_dy_emb)

            optimizer.step(trainable_layers)

        logger.info("Training loop finished.")

        text = ["hell", "worl"]
        x = vectorizer.encode(text)
        y_emb = embedding_layer.forward(x)
        h_t_final = recurrent_layer.forward(y_emb)
        p = dense_layer.forward(h_t_final)
        chosen_char_id = np.argmax(p, axis=1, keepdims=True)
        char = vectorizer.decode(np.array(chosen_char_id).reshape(-1, 1))
        print(char)
    except ValueError as e:
        logger.error("Error during network operation: %s.", e, exc_info=True)
    except RuntimeError as e:
        logger.error("Runtime error during network operation: %s.",
                     e,
                     exc_info=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An unexpected error occured: %s.", e, exc_info=True)


if __name__ == "__main__":
    main()
