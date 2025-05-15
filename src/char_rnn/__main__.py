#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

import numpy as np

from char_rnn.preprocessing import Embedding, TextVectorizer
from char_rnn.layers import Recurrent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    vectorizer = TextVectorizer()
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    vectorizer.fit(alphabet)

    texts = ["abc", "xyz"]
    indices = vectorizer.encode(texts)

    embedding_layer = Embedding(vocab_size=len(vectorizer.vocab), embed_dim=2)
    embeddings = embedding_layer.forward(indices)

    recurrent_layer = Recurrent(input_dim=2, hidden_dim=5)
    hidden_states = recurrent_layer.forward_step(embeddings[:, 0])
    for t in range(embeddings.shape[1] - 1):
        hidden_states = recurrent_layer.forward_step(embeddings[:, t + 1],
                                                     hidden_states)

    print(hidden_states)
    recurrent_layer_gradients = recurrent_layer.backward(hidden_states)
    embedding_layer.backward(recurrent_layer_gradients)


if __name__ == "__main__":
    main()
