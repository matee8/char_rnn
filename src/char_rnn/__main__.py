#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

from char_rnn.loss import SparseCategoricalCrossEntropy
from char_rnn.model import CharRNN
from char_rnn.optimizers import Adam
from char_rnn.preprocessing import TextVectorizer
from char_rnn import util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def main():
    try:
        vectorizer = TextVectorizer()
        corpus = "hello world"
        vectorizer.fit(corpus)
        V = vectorizer.vocabulary_size

        sequence = vectorizer.encode([corpus]).ravel()

        X_train, y_train = util.create_sliding_windows(sequence,
                                                       window_size=5,
                                                       shuffle=False)
        print(X_train.shape)
        print(y_train.shape)

    except ValueError as e:
        logger.error("Error during data preprocessing: %s", e, exc_info=True)
        return
    except RuntimeError as e:
        logger.error("Runtime error during data preprocessing: %s",
                     e,
                     exc_info=True)
        return

    D_e = 16
    D_h = 128
    learning_rate = 0.001

    try:
        model = CharRNN(V, D_e, D_h)

        optimizer = Adam(learning_rate)
        loss_fn = SparseCategoricalCrossEntropy()

        model.compile(optimizer, loss_fn)

        num_epochs = 1000

        for epoch in range(num_epochs):
            loss = model.train_step(X_train, y_train)

            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d - Loss: %.4f.", epoch + 1, num_epochs,
                            loss)

        logger.info("Training loop finished.")

        start_seed = "hel"
        num_to_generate = 2

        generated_text = model.generate_sequence(vectorizer, start_seed,
                                                 num_to_generate, 1.0)
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
