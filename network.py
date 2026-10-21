import numpy as np
from activations import sigmoid, tanh, relu, leaky_relu, linear, softmax , sigmoid_derivative, tanh_derivative, relu_derivative, leaky_relu_derivative, linear_derivative

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "linear": linear,
    "softmax": softmax,
}

DERIVATIVES = {
    "sigmoid":    lambda z, a: sigmoid_derivative(a),
    "tanh":       lambda z, a: tanh_derivative(a),
    "relu":       lambda z, a: relu_derivative(z),
    "leaky_relu": lambda z, a: leaky_relu_derivative(z),
    "linear":     lambda z, a: linear_derivative(z),
}

def init_params(layer_sizes, activations, seed=42):
    rng = np.random.default_rng(seed)
    W, b = [], []
    for l in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
        if activations[l] in ("relu", "leaky_relu"):
            std = np.sqrt(2 / n_in)   
        else:
            std = np.sqrt(1 / n_in)    
        W.append(rng.normal(0, std, size=(n_in, n_out)))
        b.append(np.zeros((1, n_out)))
    return W, b

def forward(X, W, b, activations):
    A = [X]
    Z = []
    a = X
    for l in range(len(W)):
        z = a @ W[l] + b[l]
        a = ACTIVATIONS[activations[l]](z)
        Z.append(z)
        A.append(a)
    return a, Z, A

def mse_loss(y_pred , y):
    return np.mean((y_pred - y) ** 2)

def bce_loss(y_pred , y):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def cce_loss(y_pred , y):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))

def backward(y, W, activations, Z, A, loss_name):
    m = y.shape[0]
    L = len(W)
    out_act = activations[-1]

    if (out_act == "softmax" and loss_name == "cce") or (
        out_act == "sigmoid" and loss_name == "bce"
    ):
        dZ = (A[-1] - y) / m
    else:  # mse
        dA = 2 * (A[-1] - y) / y.size
        dZ = dA * DERIVATIVES[out_act](Z[-1], A[-1])

    dW = [None] * L
    db = [None] * L
    for l in reversed(range(L)):
        dW[l] = A[l].T @ dZ
        db[l] = dZ.sum(axis=0, keepdims=True)
        if l > 0:
            dA = dZ @ W[l].T
            dZ = dA * DERIVATIVES[activations[l - 1]](Z[l - 1], A[l])
    return dW, db

rng = np.random.default_rng(1)
X = rng.normal(size=(6, 4))
y = np.eye(3)[rng.integers(0, 3, 6)]
acts = ["relu", "tanh", "softmax"]
W, b = init_params([4, 5, 4, 3], acts, seed=1)
out, Z, A = forward(X, W, b, acts)
dW, db = backward(y, W, acts, Z, A, "cce")

def init_adam(W, b):
    return {
        "mW": [np.zeros_like(w) for w in W],
        "vW": [np.zeros_like(w) for w in W],
        "mb": [np.zeros_like(x) for x in b],
        "vb": [np.zeros_like(x) for x in b],
        "t": 0,
    }


def adam_update(W, b, dW, db, state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for l in range(len(W)):
        for p, g, m, v in (
            (W, dW, state["mW"], state["vW"]),
            (b, db, state["mb"], state["vb"]),
        ):
            m[l] = beta1 * m[l] + (1 - beta1) * g[l]
            v[l] = beta2 * v[l] + (1 - beta2) * g[l] ** 2
            m_hat = m[l] / (1 - beta1 ** t)
            v_hat = v[l] / (1 - beta2 ** t)
            p[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)

LOSSES = {"mse": mse_loss, "bce": bce_loss, "cce": cce_loss}


def predict(X, W, b, activations):
    return forward(X, W, b, activations)[0]


def score(y_pred, y, loss_name):
    if loss_name == "cce":
        return np.mean(y_pred.argmax(axis=1) == y.argmax(axis=1))
    if loss_name == "bce":
        return np.mean((y_pred > 0.5).astype(float) == y)
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean(axis=0)) ** 2)   # R2


def train(X, y, layer_sizes, activations, loss_name, epochs=100, batch_size=32,
          lr=0.01, beta1=0.9, beta2=0.999, seed=42, X_val=None, y_val=None):
    rng = np.random.default_rng(seed)
    W, b = init_params(layer_sizes, activations, seed)
    state = init_adam(W, b)
    history = {"loss": [], "score": [], "val_loss": [], "val_score": []}
    m = X.shape[0]
    batch_size = min(batch_size, m)

    for epoch in range(epochs):
        idx = rng.permutation(m)                      # har epoch mein data shuffle
        for start in range(0, m, batch_size):
            batch = idx[start:start + batch_size]
            out, Z, A = forward(X[batch], W, b, activations)
            dW, db = backward(y[batch], W, activations, Z, A, loss_name)
            adam_update(W, b, dW, db, state, lr, beta1, beta2)

        y_pred = predict(X, W, b, activations)
        history["loss"].append(LOSSES[loss_name](y_pred, y))
        history["score"].append(score(y_pred, y, loss_name))
        if X_val is not None:
            v_pred = predict(X_val, W, b, activations)
            history["val_loss"].append(LOSSES[loss_name](v_pred, y_val))
            history["val_score"].append(score(v_pred, y_val, loss_name))
    return W, b, history

