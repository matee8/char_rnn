#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging
from typing import List

from char_rnn.preprocessing import TextVectorizer
from char_rnn.layers import Dense, Embedding, Recurrent
from char_rnn.loss import SparseCategoricalCrossEntropy

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

        x_indices = vectorizer.encode(texts)
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
    V = vectorizer.vocabulary_size

    try:
        embedding_layer = Embedding(V, D_e)
        recurrent_layer = Recurrent(D_e, D_h)
        dense_layer = Dense(D_h, V)
        loss = SparseCategoricalCrossEntropy()

        y_emb = embedding_layer.forward(x_indices[:, :-1])

        h_t_final = recurrent_layer.forward(y_emb)

        p = dense_layer.forward(h_t_final)

        mean_loss = loss.forward(p, x_indices[:, -1])

        logger.info("Mean loss after 1 pass: %.4f.", mean_loss)

        dL_dp = loss.backward(p, x_indices[:, -1])

        dL_dh_t_final = dense_layer.backward(dL_dp)

        dL_dy_emb = recurrent_layer.backward(dL_dh_t_final)

        embedding_layer.backward(dL_dy_emb)

        logger.info("Forward and backward pass completed.")
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
