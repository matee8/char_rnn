# Character-Level Recurrent Neural Network

This project implements a recurrent neural network (RNN) from scratch for
character-level text generation. This repository serves as an educational
resource for understanding the core mechanisms of neural networks, without
relying on any external deep learning framework.

# Overview

- **Custom RNN implementation**: hand-crafted neural network layers including:

    - `Embedding` layer for character-to-vector mapping.
    - `Recurrent` layer (simple RNN cell with `tanh` activation).
    - `Dense` layer with `softmax` for output probabilites.

- **Optimization**: `Adam` optimizer implementation.
- **Loss function**: `SparseCategoricalCrossEntropy` for sequence prediction.
- **Data preprocessing**:

    - `TextVectorizer` for character-to-ID mapping and vice-versa.
    - Utility for creating sliding window sequences for training.

- **Comprehensive Command-Line Interface**:

    - **`train.py`**: manages model training, including automatic dataset download, hyperparameter configuration.
    - **`inference.py`**: runs text generation from pretrained models, with temperature-controlled sampligng.

# Getting Started

## Prerequisites

- Python 3.8+
- `pip`

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/matee8/char_rnn.git
    cd char_rnn
    ```

2. **Create a virtual environment (optional but recommended)**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # or, on Windows: .\.venv\Scripts\activate
    ```

3. **Install the project**:

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
│   ├── inference.py          # Script for generating text from a trained model
│   └── train.py              # Script for training the CharRNN model
├── src/
│   └── char_rnn/
│       ├── __init__.py
│       ├── layers.py         # Neural network layer implementations
│       ├── losses.py         # Loss function implementations
│       ├── models.py         # CharRNN model architecture
│       ├── optimizers.py     # Optimization algorithms
│       ├── preprocessing.py  # Text vectorization and batching utilities
│       └── utils.py          # Utilities for data loading and model saving/loading
├── LICENSE                   # Project licensing information
├── README.md
└── requirements.txt          # Lists project dependencies for exact version pinning
```

# Usage

## Training the model

The `train.py` script is used to train the model. It will automatically download
the Tiny Shakespeare dataset if not found locally.

**Parameters**:

```plaintext
usage: train.py [-h] [--data-url DATA_URL] [--data-dir DATA_DIR] [--data-filename DATA_FILENAME] [--embedding-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM] [--learning-rate LEARNING_RATE]
                [--window-size WINDOW_SIZE] [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--seed SEED] [--model-dir MODEL_DIR] [--model-filename MODEL_FILENAME]
                [--log-interval LOG_INTERVAL]

Train a Char-RNN model.

options:
  -h, --help            show this help message and exit

Data configuration:
  --data-url DATA_URL   URL to download the dataset from. (default: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  --data-dir DATA_DIR   Directory to store downloaded data. (default: ~/Development/char_rnn/data/raw)
  --data-filename DATA_FILENAME
                        Filename for the dataset within data_dir. (default: input.txt)

Model hyperparameters:
  --embedding-dim EMBEDDING_DIM
                        Dimension of character embeddings. (default: 128)
  --hidden-dim HIDDEN_DIM
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

Output and logging configuration:
  --model-dir MODEL_DIR
                        Directory to save trained models. (default: ~/Development/char_rnn/models)
  --model-filename MODEL_FILENAME
                        Filename for the model within model_dir. (default: char_rnn_shakespeare)
  --log-interval LOG_INTERVAL
                        Log training loss every N batches. (default: 10)
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
                        Path to the model weights file (.npz). (default: ~/Development/char_rnn/models/char_rnn_shakespeare.npz)
  --vocab-data-path VOCAB_DATA_PATH
                        Path to the original text data file used to build the vocabulary. (default: ~/Development/char_rnn/data/raw/input.txt)
  --num-to-generate NUM_TO_GENERATE
                        Number of additional characters to generate. (default: 64)
  --embedding-dim EMBEDDING_DIM
                        Embedding dimension. (default: 128)
  --hidden-dim HIDDEN_DIM
                        Hidden layer dimension. (default: 10)
  --temperature TEMPERATURE
                        Temperature for sampling. (default: 1.0)
```

# Code structure

The project is structured to separate concerns.

- **`char_rnn/layers.py`**: Defines the building blocks of the neural network, abstracting operations such as forward and backward passes for layers.
- **`char_rnn/losses.py`**: Provides the interface for quantifying model prediction errors and deriving the initial gradients for backpropagation.
- **`char_rnn/models.py`**: Defines the overarching models. It manages end-to-end forward pass, loss calculation, backward pass, and parameter optimization.
- **`char_rnn/optimizers.py`**: Defines the optimizers responsible for updating model parameters based on pre-computed gradients.
- **`char_rnn/preprocessing.py`**: Offers utilities for data preparation, e.g, character encoding/decoding, creating batch sequences.
- **`char_rnn/utils.py`**: Centralizes file I/O operations for loading raw data and saving/loading model weights.

# Future enhancements

- **Validation during training**: Divide the data into a training and validation set, and log validation loss.
- **Test after training**: Evaluate the accuracy of next character prediction after training the model.
- **Module for activation functions**: Implement a module for collecting all activation functions and pass them to the layers when instantiated.
- **Advanced recurrent cells**: Implement more sophisticated RNN cells for improved sequence modeling capabilities.
- **Hyperparameter optimization**: Implement routines for automated hyperparameter tuning to find optimal model configurations.

# License

This project is licensed under the [MIT License](LICENSE).

# Acknowledgements

Inspired by Andrej Karpathy's blog post "The Unreasonable Effectiveness of Recurrent Neural Networks".
