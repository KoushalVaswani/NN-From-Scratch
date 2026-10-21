import numpy as np

def relu(x):
    return np.maximum(0 , x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(a):
    return 1 - a ** 2

def leaky_relu(x , alpha = 0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x , alpha = 0.01):
    return np.where(x > 0, 1, alpha)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True) 
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

