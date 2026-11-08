# Neural Network From Scratch

A neural network written by hand in NumPy (no TensorFlow or PyTorch) with an interactive Streamlit app.
Choose the number of hidden layers, the neurons per layer and a different activation for each layer,
then train the network and inspect what it learned.

**Live app:** _add your Streamlit Cloud link here_

![App screenshot](screenshots/app.png)

## Features
- Forward pass, backpropagation and the Adam optimiser, all written in NumPy
- Activations: sigmoid, tanh, ReLU, Leaky ReLU, linear, softmax
- Losses: binary and categorical cross-entropy (the loss is chosen from the task)
- Mini-batch training with configurable epochs, batch size and learning rate
- Built-in datasets (XOR, Moons, Circles, Iris, Digits) or upload your own CSV
  (missing values are filled, text columns are one-hot encoded, all statistics come from the training split only)
- Learning curves, network diagram, decision boundary, weight inspector
- Confusion matrix, precision, recall and F1 per class
- Predict on your own input values
- Backpropagation checked against numerical gradients (see `test_gradients.py`)

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

If `streamlit` is blocked on your machine, use `python -m streamlit run app.py`.

## Tests
    python test_gradients.py
    python test_metrics.py

## Project structure
- `activations.py`: activation functions and their derivatives
- `network.py`: initialisation, forward pass, losses, backward pass, Adam, training loop
- `preprocessing.py`: missing-value filling, scaling and one-hot encoding
- `metrics.py`: confusion matrix, precision, recall, F1
- `app.py`: Streamlit app
- `test_gradients.py`, `test_metrics.py`: tests

## Notes
The network itself is pure NumPy. scikit-learn is used only for the sample datasets, the train/test split
and the standard scaler.