#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

import numpy as np

from char_rnn.preprocessing import TextVectorizer
from char_rnn.layers import Embedding, Recurrent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    vectorizer = TextVectorizer()
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    vectorizer.fit(alphabet)
    texts = ["hello", "world"]
    indices = vectorizer.encode(texts)

    embedding_layer = Embedding(vocab_size=len(vectorizer.vocab), embed_dim=5)
    embeddings = embedding_layer.forward(indices)

    recurrent_layer = Recurrent(input_dim=5, hidden_dim=20)
    final_hidden_states = recurrent_layer.forward(embeddings)

    dummy_final_gradients = np.random.randn(*final_hidden_states.shape) * 1
    recurrent_layer_input_gradients = recurrent_layer.backward(
        dummy_final_gradients)
    print(embedding_layer.backward(recurrent_layer_input_gradients))


if __name__ == "__main__":
    main()
