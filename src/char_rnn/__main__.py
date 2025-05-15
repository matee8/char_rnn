#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

from char_rnn.preprocessing import TextVectorizer
from char_rnn.layers.embedding import Embedding

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    vectorizer = TextVectorizer()
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    vectorizer.fit(alphabet)
    texts = ["abc", "xyz"]
    indices = vectorizer.encode(texts)

    embedding_layer = Embedding(vocab_size=len(vectorizer.vocab), embed_dim=5)
    embeddings = embedding_layer.forward(indices)

    print(embeddings)


if __name__ == "__main__":
    main()
