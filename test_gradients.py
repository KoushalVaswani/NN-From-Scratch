"""Numerical gradient check: backward() must match finite differences of the loss."""
import numpy as np

from network import LOSSES, backward, forward, init_params


def numerical_grad(params, loss_fn, eps=1e-5):
    grads = []
    for P in params:
        g = np.zeros_like(P)
        for idx in np.ndindex(*P.shape):
            old = P[idx]
            P[idx] = old + eps
            up = loss_fn()
            P[idx] = old - eps
            down = loss_fn()
            P[idx] = old
            g[idx] = (up - down) / (2 * eps)
        grads.append(g)
    return grads


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))


def check(layer_sizes, activations, loss_name, one_hot):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, layer_sizes[0]))
    labels = rng.integers(0, layer_sizes[-1], 8)
    y = np.eye(layer_sizes[-1])[labels] if one_hot else rng.integers(0, 2, (8, 1)).astype(float)
    W, b = init_params(layer_sizes, activations, seed=1)

    out, Z, A = forward(X, W, b, activations)
    dW, db = backward(y, W, activations, Z, A, loss_name)

    loss_fn = lambda: LOSSES[loss_name](forward(X, W, b, activations)[0], y)
    nW, nb = numerical_grad(W, loss_fn), numerical_grad(b, loss_fn)
    err = max(max(rel_error(a, n) for a, n in zip(dW, nW)), max(rel_error(a, n) for a, n in zip(db, nb)))
    return err


CASES = [
    ([4, 5, 4, 3], ["relu", "tanh", "softmax"], "cce", True),
    ([4, 6, 3], ["leaky_relu", "softmax"], "cce", True),
    ([3, 5, 1], ["tanh", "sigmoid"], "bce", False),
    ([3, 4, 4, 1], ["sigmoid", "relu", "sigmoid"], "bce", False),
    ([3, 4, 2], ["linear", "sigmoid"], "mse", True),
]

if __name__ == "__main__":
    for sizes, acts, loss, oh in CASES:
        err = check(sizes, acts, loss, oh)
        print(f"{' -> '.join(acts):32s} {loss:4s} max relative error = {err:.2e}")
        assert err < 1e-6, "backward() does not match the numerical gradient"
    print("gradients OK")