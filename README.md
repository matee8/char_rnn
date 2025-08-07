# Character-Level Recurrent Neural Network
asd

This project implements a recurrent neural network (RNN) from scratch for
character-level text generation. This repository serves as an educational
resource for understanding the core mechanisms of neural networks, without
relying on any external deep learning framework.

# Quick links

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

# Overview

This implementation features a sequential neural network architecture designed
for character-level sequence modelling. The model processes input character
sequences to predict the next character in the sequence.

The core `Model` class is generic, allowing for the construction of various
sequential architectures by providing a list of layers. For character
generation, a helper function `create_char_rnn_model` is provided to instantiate
a common CharRNN architecture. This standard CharRNN's operation can be
conceptualized in several stages:

-   **Input representation**: Raw character sequences are first transformed
    into a dense, continuous numerical format via an **embedding layer**. The
    weights for this layer are initialized using a random uniform distribution
    by default.
-   **Sequential processing**: The embedded sequence is processed by a recurrent
    layer. The default `create_char_rnn_model` uses:

    -   A **Gated Recurrent Unit (GRU) Layer**: This layer implements the GRU
        cell, which uses update and reset gates to manage information flow,
        helping to capture longer-range dependencies. By default, it uses a tanh
        activation for the candidate hidden state and sigmoid activations for
        the gates. Input-related kernel weights are initialized with Glorot
        Uniform, and hidden-state-related recurrent weights with an Orthogonal
        initializer.

    While the default uses GRU, the generic `Model` class could also incorporate
    a Simple Recurrent Layer or other custom recurrent layers.

-   **Output prediction**: The final hidden state from the recurrent layer is
    projected through an **output layer** (a Dense layer) which uses a softmax
    activation function by default. Uses the Glorot Uniform initializer.

**Learning and Evaluation**:

-   **Loss function**: During training, the discrepancy between the model's
    predicted character probabilites and the true next character is
    quantified using a **sparse categorical cross-entropy loss function**.
-   **Optimization**: The model's internal parameters are iteratively
    adjusted to minimize this loss. The project supports multiple optimization
    algorithms, however, by default, the Nadam optimizer is used.
-   **Performance metrics**: The model's predictive capability is evaluated
    using **accuracy**, which measures the proportion of correctly predicted
    next characters on a given dataset.
-   **Statefulness**: The model processes mini-batches independently. When
    generating text sequentially, the hidden state is typically carried over
    from one step to the next. During batched training/evaluation, the hidden
    state is usually re-initialized for each batch unless explicitly managed
    otherwise.

**Text generation**:

Once trained, the model can generate new text sequences. The process involves:

-   **Initialization**: Text generation starts with a start string provided by
    the user. This string is encoded into its numerical representation. The
    initial hidden state of the recurrent layer is typically set to zeros.
-   **Iterative Precition**:

    1.  The model takes the entire sequence of characters generated so far
        (beginning with the user's start string) as input.
    2.  It processes this full sequence to predict the probability distribution
        for the next character. The recurrent layer's hidden state is internally
        re-initialized (e.g., to zeros if not otherwise specified in the layer's
        forward pass when no initial state is provided) and recomputed over the
        current sequence at each prediction step.
    3.  A new character is sampled from the predicted probability distribution.
        The temperature parameter controls the randomness of this sampling:

        -   Low temperature (e.g., < 1.0) makes the output more deterministic
            and focused, often picking the most likely characters.
        -   Temperature of 1.0 means sampling according to the model's raw
            probabilities.
        -   High temperature (e.g., 1.0 <) introduces more randomness, leading
            to more surprising or creative (but potentially less coherent) text.

    4.  The newly generated character becomes the input for the next prediction
        step.

-   **Continuation**: This iterative process is repeated for a desired number of
    characters.
-   **Output**: The sequence of generated character IDs is decoded back into
    human-readable text.

This architecture forms a complete pipeline for learning from sequential data,
from raw text input to character-level probabilistic predictions, and is
capable of generating coherent text based on the patterns it has learned.

# Getting Started

## Prerequisites

- Python 3.8+
- `pip`

## Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/matee8/char_rnn.git
    cd char_rnn
    ```

2. Create a virtual environment (optional but recommended).

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # or, on Windows: .\.venv\Scripts\activate
    ```

3. Install the project.

    The project includes a `setup.py` file, enabling installation in editable
    mode. This is the recommended approach for development, as source code
    changes will be immediately reflexted without reinstallation.

    ```bash
    pip install -e .
    ```

    Alternatively, for standard deployment, you can install direct dependencies
    from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

## Project structure

```
.
├── data/                     # Directory for raw text data (created automatically)
├── models/                   # Directory for trained model weights (created automatically)
├── scripts/
│   ├── evaluate.py           # Script for evaluating a trained model on a test set
│   ├── inference.py          # Script for generating text from a trained model
│   └── train.py              # Script for training the CharRNN model
├── src/
│   └── char_rnn/
│       ├── __init__.py
│       ├── activations.py    # Activation function implementations
│       ├── data.py           # Utilities for data loading and acquisition
│       ├── initializers.py   # Weight initializer implementations
│       ├── layers.py         # Neural network layer implementations
│       ├── losses.py         # Loss function implementations
│       ├── models.py         # Generic Model class and CharRNN factory function
│       ├── optimizers.py     # Optimization algorithms
│       └── preprocessing.py  # Text vectorization and batching utilities
├── LICENSE                   # Project licensing information
├── README.md
└── requirements.txt          # Lists project dependencies for exact version pinning
```

# Usage

## Training the model

The `train.py` script is used to train the model.

**Parameters**:

```plaintext
usage: train.py [-h] [--data-url DATA_URL] [--data-dir DATA_DIR] [--data-filename DATA_FILENAME] [--embedding-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM] [--learning-rate LEARNING_RATE]
                [--window-size WINDOW_SIZE] [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--seed SEED] [--train-size TRAIN_SIZE] [--validation-size VALIDATION_SIZE]
                [--model-dir MODEL_DIR] [--model-filename MODEL_FILENAME] [--log-interval LOG_INTERVAL]

Train a Char-RNN model.

options:
  -h, --help            show this help message and exit

Data configuration:
  --data-url DATA_URL   URL to download the dataset from. (default: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  --data-dir DATA_DIR   Directory to store downloaded data. (default: ./data/raw)
  --data-filename DATA_FILENAME
                        Filename for the dataset within data_dir. (default: input.txt)

Model hyperparameters:
  --embedding-dim EMBEDDING_DIM
                        Dimension of character embeddings. (default: 16)
  --hidden-dim HIDDEN_DIM
                        Dimension of hidden states. (default: 128)
  --learning-rate LEARNING_RATE
                        Learning rate for the Adam optimizer. (default: 0.001)

Training parameters:
  --window-size WINDOW_SIZE
                        Length of the windows passed to the RNN. (default: 100)
  --batch-size BATCH_SIZE
                        Number of sequences per batch. (default: 32)
  --num-epochs NUM_EPOCHS
                        Number of training epochs. (default: 10)
  --seed SEED           Random seed. (default: 42)
  --train-size TRAIN_SIZE
                        Proportion of the dataset to include in the training split. (default: 0.8)
  --validation-size VALIDATION_SIZE
                        Proportion of the training dataset toinclude in the validation split. (default: 0.2)

Output and logging configuration:
  --model-dir MODEL_DIR
                        Directory to save trained models. (default: ./models)
  --model-filename MODEL_FILENAME
                        Filename for the model within model_dir. (default: char_rnn_shakespeare)
  --log-interval LOG_INTERVAL
                        Log training loss every N batches. (default: 10)
```

## Evaluating the model

The `evaluate.py` script assesses the accuracy of a pre-trained CharRNN model on
a test dataset.

**Important**: The `embedding-dim` and `hidden-dim` parameters **must match**
those used during the training of the weights you are loading. The
`vocab-data-path` must point to the exact same text file used to build the
vocabulary during training to ensure consistent character-to-ID mappings.

**Parameters**:

```plaintext
usage: evaluate.py [-h] [--weights-path WEIGHTS_PATH] [--vocab-data-path VOCAB_DATA_PATH] [--embedding-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM] [--window-size WINDOW_SIZE]
                   [--batch-size BATCH_SIZE] [--seed SEED] [--test-size TEST_SIZE]

Run inference on a pre-trained CharRNN model for text generation.

options:
  -h, --help            show this help message and exit
  --weights-path WEIGHTS_PATH
                        Path to the model weights file (.npz). (default: ./char_rnn_shakespeare.npz)
  --vocab-data-path VOCAB_DATA_PATH
                        Path to the original text data file used to build the vocabulary. (default: ./data/raw/input.txt)
  --embedding-dim EMBEDDING_DIM
                        Dimension of character embeddings. (default: 16)
  --hidden-dim HIDDEN_DIM
                        Dimension of hidden states. (default: 128)
  --window-size WINDOW_SIZE
                        Length of the windows passed to the RNN. (default: 100)
  --batch-size BATCH_SIZE
                        Number of sequences per batch. (default: 32)
  --seed SEED           Random seed. (default: 42)
  --test-size TEST_SIZE
                        Proportion of the dataset to include in the testing split. (default: 0.2)
```

## Generating text (inference)

The `inference.py` script uses a pre-trained model to generate new text.

**Important**: The `embedding-dim` and `hidden-dim` parameters **must match**
those used during the training of the weights you are loading. The
`vocab-data-path` must point to the exact same text file used to build the
vocabulary during training to ensure consistent character-to-ID mappings.

**Parameters**:

```plaintext
usage: inference.py [-h] [--weights-path WEIGHTS_PATH] [--vocab-data-path VOCAB_DATA_PATH] [--num-to-generate NUM_TO_GENERATE] [--embedding-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM]
                    [--temperature TEMPERATURE]
                    start_string

Run inference on a pre-trained CharRNN model for text generation.

positional arguments:
  start_string          Initial string to start the text generation.

options:
  -h, --help            show this help message and exit
  --weights-path WEIGHTS_PATH
                        Path to the model weights file (.npz). (default: ./models/char_rnn_shakespeare.npz)
  --vocab-data-path VOCAB_DATA_PATH
                        Path to the original text data file used to build the vocabulary. (default: ./data/raw/input.txt)
  --num-to-generate NUM_TO_GENERATE
                        Number of additional characters to generate. (default: 64)
  --embedding-dim EMBEDDING_DIM
                        Dimension of character embeddings. (default: 16)
  --hidden-dim HIDDEN_DIM
                        Dimension of hidden states. (default: 128)
  --temperature TEMPERATURE
                        Temperature for sampling. (default: 1.0)
```

# Code Structure

The project is structured to separate concerns.

-   **`char_rnn/activations.py`**: Defines the activation functions which could
    be used in the layers, with their derivatives.
-   **`char_rnn/data.py`**: Contains utilities for data loading and data
    acquisition.
-   **`char_rnn/initializers.py`**: Implements various weight initialization
    strategies that can be used by layers to set their initial parameter values.
-   **`char_rnn/layers.py`**: Defines the building blocks of the neural network,
    abstracting operations such as forward and backward passes for layers.
-   **`char_rnn/losses.py`**: Provides the interface for quantifying model
    prediction errors and deriving the initial gradients for backpropagation.
-   **`char_rnn/models.py`**: Defines the overarching models. It manages
    end-to-end forward pass, loss calculation, backward pass, and parameter
    optimization.
-   **`char_rnn/optimizers.py`**: Defines the optimizers responsible for
    updating model parameters based on pre-computed gradients.
-   **`char_rnn/preprocessing.py`**: Offers utilities for data preparation, e.g,
    character encoding/decoding, creating sliding windows, batching, shuffling.

# Future Enhancements

-   **Hyperparameter optimization**: Implement routines for automated
    hyperparameter tuning to find optimal model configurations.

# License

This project is licensed under the [MIT License](LICENSE).

# Acknowledgements

Inspired by [Andrej Karpathy's blog post "The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
