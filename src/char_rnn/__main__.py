#!/usr/bin/env python3

# pylint: disable=invalid-name

import logging

from char_rnn.preprocessing import TextVectorizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    vectorizer = TextVectorizer()
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    vectorizer.fit(alphabet)

    texts = ["abc", "xyz"]
    indices = vectorizer.encode(texts)

    print(indices)


if __name__ == "__main__":
    main()
