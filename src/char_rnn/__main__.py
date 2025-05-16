#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

import numpy as np

from char_rnn.preprocessing import TextVectorizer
from char_rnn.layers import Dense, Embedding, Recurrent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def main():
    try:
        vectorizer = TextVectorizer()
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        vectorizer.fit(alphabet)
        texts = ["hello", "world"]
        indices = vectorizer.encode(texts)

        embedding_layer = Embedding(vocab_size=len(vectorizer.vocab),
                                    embed_dim=5)
        embeddings = embedding_layer.forward(indices)

        recurrent_layer = Recurrent(input_dim=5, hidden_dim=20)
        final_hidden_states = recurrent_layer.forward(embeddings)

        dense_layer = Dense(input_dim=20, output_dim=20)
        probas = dense_layer.forward(final_hidden_states)

        dummy_final_gradients = np.random.randn(*probas.shape) * 1
        dense_layer_input_gradients = dense_layer.backward(
            dummy_final_gradients)
        recurrent_layer_input_gradients = recurrent_layer.backward(
            dense_layer_input_gradients)
        print(embedding_layer.backward(recurrent_layer_input_gradients))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
