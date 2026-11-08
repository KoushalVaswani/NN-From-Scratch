# Neural Network From Scratch

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![NumPy](https://img.shields.io/badge/core-NumPy%20only-informational) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

A neural network written by hand in NumPy (no TensorFlow, no PyTorch) with an interactive Streamlit app.
Pick the hidden layers, the neurons in each layer and a different activation for every layer, then train the
network and look inside it.

## Live demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nn-from-scratch.streamlit.app/)

**Try it here: https://nn-from-scratch.streamlit.app/**

The app may take a few seconds to wake up if nobody has opened it for a while.

![App screenshot](screenshots/app.png)

| Network diagram | Decision boundary |
|---|---|
| ![Network diagram](screenshots/network.png) | ![Decision boundary](screenshots/boundary.png) |

## What you can do in the app
1. Choose a dataset (XOR, Moons, Circles, Iris, Digits) or upload your own CSV.
2. Set the number of hidden layers, neurons per layer and an activation for each layer.
3. Set epochs, batch size and learning rate, then press **Train model**.
4. Inspect the result: learning curves, confusion matrix, precision / recall / F1, the network diagram,
   the decision boundary, the weights of every layer, and a **Try it** tab to predict your own input.

## Features
- Forward pass, backpropagation and the **Adam** optimiser, all in NumPy
- Activations: sigmoid, tanh, ReLU, Leaky ReLU, linear, softmax
- Losses: binary and categorical cross-entropy, chosen automatically from the task
- He / Xavier weight initialisation, mini-batch training
- CSV upload with automatic preprocessing: missing values filled, text columns one-hot encoded,
  numbers standardised. All statistics come from the training split only, so nothing leaks from the test set
- Stratified train / test split
- Backpropagation verified against numerical gradients (see `test_gradients.py`)

## How it works
- **Forward:** `a = activation(a_prev @ W + b)` for every layer; the output layer is sigmoid (2 classes) or softmax (more).
- **Backward:** for softmax + cross-entropy and sigmoid + binary cross-entropy the output gradient simplifies to
  `(prediction - label) / batch_size`; the error is then pushed back layer by layer using each activation's derivative.
- **Adam:** running averages of the gradient and of its square, with bias correction.
- **Check:** `test_gradients.py` compares `backward()` with finite differences on five different architectures.
  The relative error is below 1e-7.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

If the `streamlit` command is blocked on your machine, use `python -m streamlit run app.py`.

## Tests
    python test_gradients.py
    python test_metrics.py

## Project structure
    activations.py     activation functions and their derivatives
    network.py         initialisation, forward, losses, backward, Adam, training loop
    preprocessing.py   missing values, scaling, one-hot encoding
    metrics.py         confusion matrix, precision, recall, F1
    app.py             Streamlit app
    test_gradients.py  numerical gradient check
    test_metrics.py    metrics compared with scikit-learn

## Roadmap
- Early stopping and L2 / dropout regularisation
- Regression support
- Save and load trained models
- Compare several runs in one chart

## Notes
The network itself is pure NumPy. scikit-learn is used only for the sample datasets, the train / test split and
the standard scaler (and, in the tests, to check the metrics).
