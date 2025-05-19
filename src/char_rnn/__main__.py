#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging
import random
from typing import List

from char_rnn import util
from char_rnn.loss import SparseCategoricalCrossEntropy
from char_rnn.model import CharRNN
from char_rnn.optimizers import Adam
from char_rnn.preprocessing import TextVectorizer

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

D_E = 16
D_H = 128
ETA = 0.001
L_WINDOW = 5
N_BATCH = 5
NUM_EPOCHS = 1000
SHUFFLE_DATA_EACH_EPOCH = True
LOG_INTERVAL_EPOCHS = 50
SEED = 42


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    try:
        vectorizer = TextVectorizer()
        corpus = (
            "hello world this is a demonstration of a character rnn with "
            "mini batch training this corpus is also very small for good "
            "results")
        vectorizer.fit(corpus)
        V = vectorizer.vocabulary_size

        x = vectorizer.encode([corpus]).ravel()
    except ValueError as e:
        logger.error("Error during data preprocessing: %s", e, exc_info=True)
        return
    except RuntimeError as e:
        logger.error("Runtime error during data preprocessing: %s",
                     e,
                     exc_info=True)
        return

    try:
        model = CharRNN(V, D_E, D_H)

        optimizer = Adam(ETA)
        loss_fn = SparseCategoricalCrossEntropy()

        model.compile(optimizer, loss_fn)

        logger.info(
            "Starting training with %d epochs, batch_size=%d, "
            "window_size=%d.", NUM_EPOCHS, N_BATCH, L_WINDOW)

        for epoch in range(NUM_EPOCHS):
            epoch_losses: List[float] = []

            mini_batch_generator = util.create_batch_sequences(
                x,
                L_w=L_WINDOW,
                N=N_BATCH,
                shuffle=SHUFFLE_DATA_EACH_EPOCH,
                seed=SEED + epoch if SHUFFLE_DATA_EACH_EPOCH else SEED,
                drop_last=True)

            for X_batch, y_batch in mini_batch_generator:
                batch_loss = model.train_step(X_batch, y_batch)
                epoch_losses.append(batch_loss)

            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                if (epoch + 1) % LOG_INTERVAL_EPOCHS == 0:
                    logger.info("Epoch %d/%d - Loss: %.4f.", epoch + 1,
                                NUM_EPOCHS, avg_epoch_loss)
            else:
                logger.warning("Epoch %d/%d - No batches were processed.",
                               epoch + 1, NUM_EPOCHS)

        logger.info("Training loop finished.")

        prompt = "hell"
        num_to_generate = 50

        generated_text = model.generate_sequence(vectorizer=vectorizer,
                                                 prompt=prompt,
                                                 n_chars=num_to_generate,
                                                 temperature=1.0)
        logger.info("Generated text (%d chars, temp=1.0): '%s'",
                    num_to_generate, generated_text)
    except ValueError as e:
        logger.error("ValueError during model operation: %s", e, exc_info=True)
    except RuntimeError as e:
        logger.error("RuntimeError during model operation: %s",
                     e,
                     exc_info=True)
    except TypeError as e:
        logger.error("TypeError during model operation: %s", e, exc_info=True)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
