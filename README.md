# Neural Network From Scratch

A fully hand-written neural network in NumPy (no TensorFlow / PyTorch) with an interactive
Streamlit app. Choose the number of hidden layers, neurons per layer and a different
activation for each layer, then train and inspect the network.

## Features
- Forward pass, backpropagation and the Adam optimizer written from scratch in NumPy
- Activations: sigmoid, tanh, relu, leaky_relu, linear, softmax
- Losses: binary cross-entropy, categorical cross-entropy
- Mini-batch training with configurable epochs, batch size and learning rate
- Learning curves, network diagram, decision boundary plot, weight inspector
- Predict on your own input values
- Backprop verified with numerical gradient checking

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Project structure
- `activations.py`: activation functions and derivatives
- `network.py`: init, forward, loss, backward, Adam, training loop
- `app.py`: Streamlit UI